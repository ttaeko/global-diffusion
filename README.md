# Global Diffusion

![An alpine setting, ~100km x 50km](https://github.com/ttaeko/global-diffusion/blob/main/img/mountains.png)
An alpine setting ~100km x ~50km

This is a heavily modified version of [[Terrain Diffusion]](https://xandergos.github.io/terrain-diffusion/) with an explicit focus on continental scale realism.
Its intended use case is Minecraft, for which there will be a Fabric mod once the behind-the-scenes modifications are fully implemented and promoted.

## Modifications/Functionality
Fundamentally, Terrain Diffusion and Global Diffusion work the same way. Coarse terrain is gradually refined to higher resolution with the use of learned generators. 
In the case of Global Diffusion, the initial coarse stage learns to produce land masses at much lower frequencies, think multiple thousands of kilometers. This results
in truly continental scale generation, instead than the islands of Terrain Diffusion. Additionally, the final scale is higher. Global Diffusion refines the terrain down
to a resolution of 2 meters using inexpensive one-pass U-Nets. The central latent 240m stage and 30m decoder are exactly the same as Terrain Diffusion's.

Global Diffusion also introduces deterministically generated hydrology based off of the 240m physical low frequency channel present in the latent stage.
It is gradually refined throughout the downstream stages and reconciled to 30m, 10m and finally 2m, producing realistic rivers and lakes.

I will be the first to admit that this implementation is impractical, but call it a stubborn unwillingness to give up in my quest for ever more realistic terrain in Minecraft.
Or something like that.

Portions of this project were developed with the help of generative AI. This is just a casual project for my Minecraft server, Taeko & Co., inspired by my home, the Swiss Alps :)

## Attribution

This project is based on Terrain Diffusion by xandergos.
The original implementation is available [here](https://github.com/xandergos/terrain-diffusion).

Substantial modifications and additional components have been made for this
project. See the repository history for subsequent changes.

Based on upstream commit: b39ac3c, from July 21, 2026.
