# Hair with Anisotropy

A transparent hair shader for Unity/VRchat that implements smooth dithered transparency and anisotropic highlights, reacting to lighting properly. Includes handling for baked lighting with an approximated light from light probes.

![Preview](https://gitlab.com/s-ilent/hair-with-anisotropy/-/wikis/uploads/25d3859ff5c9cf8a23944c3fbce6d724/Unity_2020-11-27_21-49-59c.png)

## Installation

Download the repository. Then place the Shader/ folder with the shader into your Assets/ directory.

## Usage

This shader contains several options that gover the different aspects of the material.

* **Cutout and Transparency Threshold**<br>
  These control the cutoff point for the hair's transparency and opacity. Raising cutout will make the transparent regions of the hair fully transparent. Lowering transparency threshold will make the opaque regions fully opaque. Use this to fine tune the appearance of hair transparency. 
* **Sharp Transparency**<br>
  When this is active, transparency will always be sharpened to a 2 pixel wide gradient.

* **Use Energy Conservation**<br>
  When active, the brightness of the specular colour will darken the albedo.
* **Use Specular Colour**<br>
  By default, the colour of hair shine is derived from the Albedo texture and the Metallic slider. When active, the Specular Colour property is used directly. 
* **Metallic**<br>
  The level of light absorption in the specular. (While hair isn't actually metallic, the absorption of light is related to conductivity.)
* **Reflectivity**<br>
  Controls the strength of glossy reflections on the hair.
* **Anisotropy**<br>
  Controls the direction of anisotropic highlights on the hair. The highlight direction always lies acros the mesh's tangents/bitangents.
* **Tangent Shift**<br>
  This shifts the position of the specular highlight along the hair tangents and visually pushes it around.
* **Gloss Power**<br>
  This controls the width of the specular highlights for the two seperate highlights on the hair.
* **Tangent Shift Texture**<br>
  Allows specifying a seperate texture with a tangent shift pattern to make hair look less flat.
* **Occlusion**<br>
  Specifies areas of the hair that are less reached by light, and more shaded. 

## License?

This work is licensed under MIT license.