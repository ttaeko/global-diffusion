"""Adapter running the existing 240 m hydrology kernels on stock lowfreq terrain."""
from __future__ import annotations
from dataclasses import asdict, replace
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
from .compiled_routing import priority_flood_route_compiled, strahler_order_d8
from .hybrid_conditioning import hybrid_fill_breach_route
from .lakes import identify_depression_lakes
from .network import extract_river_graph
from .multires import build_regional_boundary_conditions
from .macro_constraints import ensure_routing_zone_terminals
from .planner import HydrologyPlannerConfig, plan_hydrology
from .profile_contract import DEFAULT_HYDROLOGY_PROFILE
from .runoff import mean_discharge_from_runoff
from .training_profile import (
    apply_hydrology_terrain_transform,
    build_hydrology_training_profile,
)

_NODATA = np.iinfo(np.uint32).max

def _divides(ids, land):
    """Existing macro_topology._catchment_divides logic."""
    result = np.zeros(land.shape, bool)
    for dr, dc in ((0,1),(1,0),(1,1),(1,-1)):
        sr=slice(max(0,-dr),min(land.shape[0],land.shape[0]-dr)); sc=slice(max(0,-dc),min(land.shape[1],land.shape[1]-dc))
        tr=slice(max(0,dr),min(land.shape[0],land.shape[0]+dr)); tc=slice(max(0,dc),min(land.shape[1],land.shape[1]+dc))
        edge=land[sr,sc]&land[tr,tc]&(ids[sr,sc]!=ids[tr,tc]); result[sr,sc]|=edge; result[tr,tc]|=edge
    return result

def _fill_codes(codes):
    missing=codes==_NODATA
    if not np.any(~missing): raise ValueError("Macro topology contains no routed basin")
    return codes[tuple(scipy.ndimage.distance_transform_edt(missing, return_distances=False, return_indices=True))].astype(np.uint32) if np.any(missing) else codes

def _snap_zones(dem, macro, refinement=32):
    """Existing macro_constraints.snap_projected_basin_zones, fixed at 16 cells."""
    zones=np.repeat(np.repeat(macro,refinement,0),refinement,1)
    boundary=np.zeros(zones.shape,bool); boundary[1:]|=zones[1:]!=zones[:-1]; boundary[:-1]|=zones[:-1]!=zones[1:]; boundary[:,1:]|=zones[:,1:]!=zones[:,:-1]; boundary[:,:-1]|=zones[:,:-1]!=zones[:,1:]
    relaxed=scipy.ndimage.binary_dilation(boundary,iterations=16)
    unique,inverse=np.unique(zones,return_inverse=True); markers=(inverse.reshape(zones.shape)+1).astype(np.int32); markers[relaxed]=0
    for marker in range(1,unique.size+1):
        if not np.any(markers==marker):
            candidates=np.argwhere(zones==unique[marker-1]); row,col=candidates[np.nanargmin(dem[candidates[:,0],candidates[:,1]])]; markers[row,col]=marker
    lo,hi=np.percentile(dem[np.isfinite(dem)],[1,99]); cost=np.asarray(np.round(np.clip((dem-lo)/max(hi-lo,1e-6),0,1)*255),np.uint8)
    watershed=scipy.ndimage.watershed_ift(cost,markers)
    return unique[np.clip(watershed-1,0,unique.size-1)].astype(np.uint32), relaxed

def _reconcile(flow, catchments, prior):
    """Existing hierarchical_planner.reconcile_basin_projection without atlas portals."""
    valid=catchments!=_NODATA; lookup=np.full(int(catchments[valid].max())+1,_NODATA,np.uint32); outlets=valid&(flow==0); lookup[catchments[outlets]]=prior[outlets]
    if np.any(lookup[catchments[valid]]==_NODATA): raise RuntimeError("A routed catchment has no basin identity anchor")
    projected=np.full(flow.shape,_NODATA,np.uint32); projected[valid]=lookup[catchments[valid]]
    return projected,_divides(projected,valid)

def _hillshade(dem):
    gy,gx=np.gradient(dem,240.,240.); slope=np.arctan(np.hypot(gx,gy)); aspect=np.arctan2(-gx,gy)
    return np.clip(np.sin(np.deg2rad(45))*np.cos(slope)+np.cos(np.deg2rad(45))*np.sin(slope)*np.cos(np.deg2rad(315)-aspect),0,1)

def _png(path, values, cmap, log=False):
    plt.imsave(path,np.log10(np.maximum(values,1)) if log else values,cmap=cmap)

def _project_macro_precipitation(macro_precipitation_mm):
    """Existing macro_world 7.68 km -> 240 m bilinear climate projection."""
    return np.maximum(scipy.ndimage.zoom(np.asarray(macro_precipitation_mm,np.float32),32,order=1,mode="nearest",prefilter=False,grid_mode=True),0).astype(np.float32)

def _write_river_graph(path, graph):
    """Persist the existing extracted 240 m vector graph in portable JSON."""
    path.write_text(json.dumps({"resolution_m":240.0,"nodes":[asdict(x) for x in graph.nodes],"edges":[asdict(x) for x in graph.edges]},indent=2)+"\n",encoding="utf-8")

def run_stock_mosaic_hydrology(output_directory, *, macro_elevation_m, macro_precipitation_mm, lowfreq_m):
    """Use lowfreq_m directly as the old 240 m terrain input; no interpolation."""
    output=Path(output_directory); macro=np.asarray(macro_elevation_m,np.float32); terrain=np.asarray(lowfreq_m,np.float32)
    macro_precipitation=np.asarray(macro_precipitation_mm,np.float32)
    if macro.ndim!=2 or macro_precipitation.shape!=macro.shape or terrain.ndim!=2 or terrain.shape != (macro.shape[0]*32,macro.shape[1]*32): raise ValueError("240 m DEM and macro precipitation must nest exactly 32x in macro crop")
    macro_land=np.isfinite(macro)&(macro>0); land=np.isfinite(terrain)&(terrain>0)
    if not macro_land.any() or not land.any():
        raise ValueError(
            "Hydrology requires land at both resolutions; "
            f"macro_land_fraction={float(macro_land.mean()):.4f}, "
            f"lowfreq_land_fraction={float(land.mean()):.4f}, "
            f"macro_range_m=[{float(np.nanmin(macro)):.1f}, {float(np.nanmax(macro)):.1f}], "
            f"lowfreq_range_m=[{float(np.nanmin(terrain)):.1f}, {float(np.nanmax(terrain)):.1f}]. "
            "Choose another seed/crop or use the normal macro schedule; do not "
            "run the 30 m and 10 m hydrology stages on an all-ocean stock DEM."
        )
    macro_route=priority_flood_route_compiled(macro,resolution_m=7680.,land_mask=macro_land,open_boundary=True)
    macro_codes=_fill_codes(macro_route.catchment_id); macro_divide=_divides(macro_route.catchment_id,macro_land)
    basin_prior,divide_relaxation=_snap_zones(terrain,macro_codes)
    continental_prior=_divides(basin_prior,land)
    initial=priority_flood_route_compiled(terrain,resolution_m=240.,land_mask=land,open_boundary=True)
    lakes=identify_depression_lakes(terrain,initial.elevation_conditioned_m,resolution_m=240.,land_mask=land,maximum_total_lake_fraction=.005)
    hybrid=hybrid_fill_breach_route(terrain,resolution_m=240.,land_mask=land,preserve_mask=lakes.lake_mask,fill_tolerance_m=10.,breach_minimum_area_km2=10.,breach_minimum_depth_m=50.,maximum_breach_incision_m=800.,passes=2)
    route=hybrid.routing; channels=(route.accumulation_area_m2>=DEFAULT_HYDROLOGY_PROFILE.channel_minimum_area_km2*1e6)&land; order=strahler_order_d8(route.flow_direction,route.processing_order,channels); basin,divide=_reconcile(route.flow_direction,route.catchment_id,basin_prior)
    precipitation=DEFAULT_HYDROLOGY_PROFILE.calibrate_generated_precipitation(_project_macro_precipitation(macro_precipitation)); precipitation[~land]=0
    discharge=mean_discharge_from_runoff(route.flow_direction,route.processing_order,precipitation,resolution_m=240.,runoff_ratio=DEFAULT_HYDROLOGY_PROFILE.runoff_ratio)
    graph=extract_river_graph(route.flow_direction,channels,hybrid.elevation_breached_m,route.accumulation_area_m2,route.catchment_id,order,resolution_m=240.,mean_discharge_m3s=discharge,lake_id=lakes.lake_id)
    _write_river_graph(output/"river_network_240m.json",graph)
    arrays={"lowfreq_m_240m.npy":terrain,"macro_elevation_m_7680m.npy":macro,"macro_annual_precipitation_mm_7680m.npy":macro_precipitation,"annual_precipitation_mm_240m.npy":precipitation,"macro_basin_code_7680m.npy":macro_codes,"macro_divide_mask_7680m.npy":macro_divide.astype(np.uint8),"basin_topology_prior_240m.npy":basin_prior,"macro_divide_relaxation_mask_240m.npy":divide_relaxation.astype(np.uint8),"continental_divide_prior_240m.npy":continental_prior.astype(np.uint8),"catchment_id_240m.npy":route.catchment_id,"basin_code_240m.npy":basin,"continental_divide_mask_240m.npy":divide.astype(np.uint8),"flow_direction_d8_240m.npy":route.flow_direction,"flow_accumulation_area_m2_240m.npy":route.accumulation_area_m2,"mean_discharge_m3s_240m.npy":discharge,"channel_mask_240m.npy":channels.astype(np.uint8),"stream_order_240m.npy":order,"lake_id_240m.npy":lakes.lake_id,"lake_mask_240m.npy":lakes.lake_mask.astype(np.uint8),"water_surface_elevation_m_240m.npy":lakes.water_surface_elevation_m,"elevation_breached_m_240m.npy":hybrid.elevation_breached_m,"elevation_conditioned_m_240m.npy":route.elevation_conditioned_m,"elevation_correction_m_240m.npy":route.elevation_correction_m,"breach_mask_240m.npy":hybrid.breach_mask.astype(np.uint8)}
    for name,value in arrays.items(): np.save(output/name,value)
    for name,value,cmap,log in (("lowfreq_m_240m.png",terrain,"terrain",False),("lowfreq_m_hillshade_240m.png",_hillshade(terrain),"gray",False),("basin_topology_prior_240m.png",basin_prior,"tab20",False),("basin_code_240m.png",basin,"tab20",False),("continental_divide_prior_240m.png",continental_prior,"gray",False),("continental_divide_mask_240m.png",divide,"gray",False),("flow_accumulation_area_240m.png",route.accumulation_area_m2,"viridis",True),("channel_mask_240m.png",channels,"Blues",False),("lake_mask_240m.png",lakes.lake_mask,"Blues",False),("breach_mask_240m.png",hybrid.breach_mask,"magma",False)): _png(output/name,value,cmap,log)
    report={"terrain_source":"stock hybrid_mosaic[4] -> LOWFREQ_STD/MEAN -> signed square (metres)","precipitation_source":"macro annual_precipitation_mm -> existing bilinear 32x projection -> existing profile calibration","macro_resolution_m":7680.,"planner_resolution_m":240.,"refinement":32,"macro_shape":list(macro.shape),"planner_shape":list(terrain.shape),"channel_minimum_area_km2":DEFAULT_HYDROLOGY_PROFILE.channel_minimum_area_km2,"runoff_ratio":DEFAULT_HYDROLOGY_PROFILE.runoff_ratio,"river_graph_edges":len(graph.edges),"lake_count":len(lakes.records),"lake_fraction_of_land":float(lakes.lake_mask.sum()/land.sum()),"breach_fraction_of_land":float(hybrid.breach_mask.sum()/land.sum()),"breach_passes":[asdict(x) for x in hybrid.metrics],"finite_mosaic_boundary":"open boundary at both levels; no atlas portals or closed-continent expansion"}
    (output/"hydrology_report.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    return report

def render_river_overlay_on_30m_hillshade(output_directory, elevation_m):
    """Draw existing 240 m vector edges on 30 m terrain; routing remains 240 m."""
    output=Path(output_directory); dem=np.asarray(elevation_m,np.float32)
    if dem.shape != (3584,3584): raise ValueError("Expected the blended 3584x3584 30 m mosaic")
    graph=json.loads((output/"river_network_240m.json").read_text(encoding="utf-8"))
    hillshade=np.clip(_hillshade_30m(dem),0,1); plt.imsave(output/"blended_30m_hillshade.png",hillshade,cmap="gray")
    figure,axis=plt.subplots(figsize=(14,14),dpi=180); axis.imshow(hillshade,cmap="gray",interpolation="nearest")
    for edge in graph["edges"]:
        points=np.asarray(edge["cells"],dtype=float); q=max(float(edge["mean_discharge_m3s"] or 0),1e-3)
        width=0.6+0.55*np.clip(np.log10(q)+2.0,0.0,4.0)
        axis.plot(points[:,1]*8+4,points[:,0]*8+4,color="#26a9ff",alpha=.82,linewidth=width,solid_capstyle="round")
    axis.axis("off"); figure.savefig(output/"blended_30m_hillshade_rivers_240m.png",bbox_inches="tight",pad_inches=0); plt.close(figure)
    (output/"river_overlay_metadata.json").write_text(json.dumps({"network":"river_network_240m.json from existing extract_river_graph","mapping":"240 m cell centre -> 30 m pixel centre: row/col * 8 + 4","inclusion_field":"channel_mask_240m.npy; existing accumulation-area threshold of 11.875 km²","line_width_field":"mean_discharge_m3s_240m.npy","stream_order_field":"stream_order_240m.npy; carried in the vector graph but not used for width","line_width":"0.6 + 0.55 * clip(log10(max(Q,1e-3)) + 2, 0, 4) display points","no_30m_rerouting":True},indent=2)+"\n",encoding="utf-8")

def _hillshade_30m(dem):
    gy,gx=np.gradient(dem,30.,30.); slope=np.arctan(np.hypot(gx,gy)); aspect=np.arctan2(-gx,gy)
    return np.sin(np.deg2rad(45))*np.cos(slope)+np.cos(np.deg2rad(45))*np.sin(slope)*np.cos(np.deg2rad(315)-aspect)

def _signed_sqrt(values):
    return np.sign(values)*np.sqrt(np.abs(values))

def _draw_graph(axis, graph, *, color, alpha=.82):
    edges=graph["edges"] if isinstance(graph,dict) else graph.edges
    for edge in edges:
        cells=edge["cells"] if isinstance(edge,dict) else edge.cells
        discharge=edge["mean_discharge_m3s"] if isinstance(edge,dict) else edge.mean_discharge_m3s
        points=np.asarray(cells,dtype=float); q=max(float(discharge or 0),1e-3)
        width=0.6+0.55*np.clip(np.log10(q)+2.0,0.0,4.0)
        axis.plot(points[:,1]+.5,points[:,0]+.5,color=color,alpha=alpha,linewidth=width,solid_capstyle="round")

def _write_30m_overlay(path, hillshade, graph, *, comparison=None):
    figure,axis=plt.subplots(figsize=(14,14),dpi=180); axis.imshow(hillshade,cmap="gray",interpolation="nearest")
    if comparison is not None:
        projected={**comparison}
        for edge in projected["edges"]:
            points=np.asarray(edge["cells"],dtype=float)
            axis.plot(points[:,1]*8+4,points[:,0]*8+4,color="#ffc928",alpha=.65,linewidth=.8,solid_capstyle="round")
    _draw_graph(axis,graph,color="#26a9ff"); axis.axis("off"); figure.savefig(path,bbox_inches="tight",pad_inches=0); plt.close(figure)

def reconcile_stock_mosaic_hydrology_30m(output_directory, elevation_m):
    """Directly apply the existing 240 m -> 30 m regional reconciliation path.

    This bypasses WorldPlanStore only: the exact existing multires boundary,
    planner, profile, transform, and vector-network functions remain in use.
    """
    output=Path(output_directory); elevation=np.asarray(elevation_m,np.float32)
    flow=np.load(output/"flow_direction_d8_240m.npy"); accumulation=np.load(output/"flow_accumulation_area_m2_240m.npy")
    catchments=np.load(output/"catchment_id_240m.npy"); lowfreq=np.load(output/"lowfreq_m_240m.npy")
    discharge=np.load(output/"mean_discharge_m3s_240m.npy"); precipitation_240=np.load(output/"annual_precipitation_mm_240m.npy")
    if elevation.shape != tuple(np.multiply(flow.shape,8)):
        raise ValueError("Blended 30 m DEM must be exactly 8x the 240 m hydrology plan")
    coarse_land=np.isfinite(lowfreq)&(lowfreq>0)
    boundary=build_regional_boundary_conditions(flow,accumulation,catchments,coarse_land,row_start=0,col_start=0,height=flow.shape[0],width=flow.shape[1],refinement=8)
    # The stock 30 m residual can locally lift a cell above zero inside a
    # 240 m ocean parent; it has no inherited catchment and must not become a
    # new, unanchored topology component. Keep the 240 m plan authoritative
    # by excluding it from regional land. Ocean-zone labels are retained,
    # since the old zoned router uses their matching coarse basin identity to
    # seed a coastal land sector.
    inherited_land=np.repeat(np.repeat(coarse_land,8,axis=0),8,axis=1)
    inherited_discharge=np.zeros(elevation.shape,np.float64)
    for portal in boundary.portals:
        if portal.kind == "inflow":
            value=discharge[portal.global_row,portal.global_col]
            if np.isfinite(value): inherited_discharge[portal.regional_row,portal.regional_col]+=value
    precipitation=np.maximum(scipy.ndimage.zoom(precipitation_240,8,order=1,mode="nearest",prefilter=False,grid_mode=True),0).astype(np.float32)
    valid_zone=boundary.routing_zones != _NODATA
    land=(elevation>0)&inherited_land&valid_zone; land|=boundary.terminal_mask; precipitation[~land]=0
    terminals,zones,zone_cleanup=ensure_routing_zone_terminals(
        elevation,land,boundary.routing_zones,boundary.terminal_mask
    )
    # This transplant permits the old cleanup's coastal-terminal repair, but
    # refuses its exceptional zone-reassignment fallback: macro ownership must
    # stay hard-authoritative at 30 m.
    if zone_cleanup["reassigned_interior_components"]:
        raise RuntimeError(
            "30 m reconciliation would reassign inherited 240 m catchment "
            "ownership; refusing to loosen the topology constraint"
        )
    boundary=replace(boundary,terminal_mask=terminals,routing_zones=zones)
    planned=plan_hydrology(elevation,resolution_m=30.,land_mask=land,precipitation_mm_year=precipitation,terminal_mask=boundary.terminal_mask,open_boundary=False,routing_zones=boundary.routing_zones,initial_accumulation_area_m2=boundary.initial_accumulation_area_m2,initial_discharge_m3s=inherited_discharge,config=HydrologyPlannerConfig(channel_minimum_area_km2=DEFAULT_HYDROLOGY_PROFILE.channel_minimum_area_km2,reference_precipitation_mm_year=DEFAULT_HYDROLOGY_PROFILE.reference_precipitation_mm_year,runoff_ratio=DEFAULT_HYDROLOGY_PROFILE.runoff_ratio,conditioning_distance_scale_m=DEFAULT_HYDROLOGY_PROFILE.conditioning_distance_scale_m))
    if planned.conditioning is None: raise RuntimeError("Existing planner did not return conditioning")
    # The stock 30 m residual can place a generated channel source below the
    # separately rasterized 30 m lake level.  Use the existing profile's
    # non-strict compatibility path: it preserves that source height and
    # disables only an impossible raise, while the transform stays strictly
    # incision-only.  This is not a new lake or channel algorithm.
    profile=build_hydrology_training_profile(_signed_sqrt(elevation),planned.conditioning.values,**DEFAULT_HYDROLOGY_PROFILE.profile_kwargs(resolution_m=30.),sea_level_elevation_m=0.,lake_water_surface_elevation_m=planned.lakes.water_surface_elevation_m,strict_outlet_floor=False)
    repaired=apply_hydrology_terrain_transform(elevation,profile)
    graph=extract_river_graph(planned.routing.flow_direction,planned.channel_mask,repaired,planned.routing.accumulation_area_m2,planned.routing.catchment_id,planned.stream_order,resolution_m=30.,mean_discharge_m3s=planned.mean_discharge_m3s,lake_id=planned.lakes.lake_id)
    _write_river_graph(output/"river_network_reconciled_30m.json",graph)
    arrays={"reconciled_30m_flow_direction_d8.npy":planned.routing.flow_direction,"reconciled_30m_accumulation_area_m2.npy":planned.routing.accumulation_area_m2,"reconciled_30m_mean_discharge_m3s.npy":planned.mean_discharge_m3s,"reconciled_30m_catchment_id.npy":planned.routing.catchment_id,"reconciled_30m_routing_zones.npy":boundary.routing_zones,"reconciled_30m_inherited_land_mask.npy":inherited_land.astype(np.uint8),"reconciled_30m_channel_mask.npy":planned.channel_mask.astype(np.uint8),"reconciled_30m_stream_order.npy":planned.stream_order,"reconciled_30m_lake_id.npy":planned.lakes.lake_id,"reconciled_30m_lake_mask.npy":planned.lakes.lake_mask.astype(np.uint8),"reconciled_30m_water_surface_elevation_m.npy":planned.lakes.water_surface_elevation_m,"reconciled_30m_elevation_conditioned_m.npy":planned.routing.elevation_conditioned_m,"reconciled_30m_hydrology_repaired_dem_m.npy":repaired,"reconciled_30m_profile_incision_m.npy":profile.terrain_correction_m,"reconciled_30m_annual_precipitation_mm.npy":precipitation}
    for name,value in arrays.items(): np.save(output/name,value)
    original_hillshade=np.clip(_hillshade_30m(elevation),0,1); repaired_hillshade=np.clip(_hillshade_30m(repaired),0,1)
    plt.imsave(output/"reconciled_30m_hydrology_repaired_hillshade.png",repaired_hillshade,cmap="gray")
    _write_30m_overlay(output/"blended_30m_hillshade_rivers_reconciled_30m.png",original_hillshade,graph)
    _write_30m_overlay(output/"reconciled_30m_hydrology_repaired_hillshade_rivers.png",repaired_hillshade,graph)
    projected=json.loads((output/"river_network_240m.json").read_text(encoding="utf-8"))
    _write_30m_overlay(output/"blended_30m_hillshade_projected_vs_reconciled_rivers.png",original_hillshade,graph,comparison=projected)
    report={"topology_source":"existing build_regional_boundary_conditions with refinement=8 and hard inherited routing_zones","world_plan_store":"bypassed; direct in-memory call of existing regional functions only","portal_count":len(boundary.portals),"inflow_portal_count":sum(x.kind=="inflow" for x in boundary.portals),"outlet_portal_count":sum(x.kind=="outlet" for x in boundary.portals),"inherited_upstream_area_m2":float(boundary.initial_accumulation_area_m2.sum()),"inherited_discharge_m3s":float(inherited_discharge.sum()),"routing_zone_terminal_cleanup":zone_cleanup,"stock_30m_land_without_240m_parent_cells":int(np.count_nonzero((elevation>0)&~inherited_land)),"profile":"existing build_hydrology_training_profile then incision-only apply_hydrology_terrain_transform","profile_strict_outlet_floor":False,"profile_compatibility":"existing source-below-lake-floor preservation; no terrain raising","no_10m_stage":True}
    (output/"reconciled_30m_hydrology_report.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    return report
