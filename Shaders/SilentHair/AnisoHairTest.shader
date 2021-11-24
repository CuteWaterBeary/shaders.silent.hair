Shader "Silent/Hair Anisotropic"
// Thanks to d4rkpl4y3r for providing the base vert/frag Standard lighting shader!
{
	Properties
	{
		[HDR] _Color("Tint", Color) = (1,1,1,1)
		_MainTex("Albedo", 2D) = "white" {}
		_Cutoff("Cutout", Range(0,1)) = 0.5
		_ClampCutoff("Transparency Threshold", Range(1, 0)) = 0.5
		[ToggleUI]_AlphaSharp("Sharp Transparency", Float) = 0.0

		[Space]
		_BumpMap("Normals", 2D) = "bump" {}
		_BumpScale("Normal Map Scale", Float) = 1
		_Emission("Emission", Range(0, 20)) = 0
		
		[Header(Specular)]
		[Toggle(FINALPASS)]_UseEnergyConserv ("Use Energy Conservation", Range(0, 1)) = 0
		[Toggle(BLOOM)]_UseSpecColor ("Use Specular Color", Range(0, 1)) = 0
		[ToggleUI]_FakeDirectional ("Fake directional light if missing", Range(0, 1)) = 0
		_SpecularColor("Specular Color", Color) = (0.5, 0.5, 0.5, 1.0)
		[Gamma] _Metallic("Metallic", Range(0, 1)) = 0
		_Smoothness("Reflectivity", Range(0, 1)) = 0
		_AnisotropyA("Anisotropy", Range(-1, 1)) = 0
		//_AnisotropyA("Anisotropy α", Range(-1, 1)) = 0
		//_AnisotropyB("Anisotropy β", Range(-1, 1)) = 0
		_TangentA("Tangent Shift A", Range(-1, 1)) = 0.5
		_TangentB("Tangent Shift B", Range(-1, 1)) = 0
		_GlossA("Gloss Power A", Range(0, 1)) = 0.6
		_GlossB("Gloss Power B", Range(0, 1)) = 1
		[Header(Advanced)]
		[Toggle(BLOOM_LOW)]_UseTangentTexture ("Use Tangent Shift Texture", Range(0, 1)) = 0
		_TangentShiftTex("Tangent Shift Texture", 2D) = "black" {}
		_OcclusionMap("Occlusion Map", 2D) = "white" {}
		_OcclusionScale("Occlusion Scale", Range(0,1)) = 1.0
		[Header(System)]
		[Enum(Off, 0, Front, 1, Back, 2)] _Culling ("Culling Mode", Int) = 2
		[ToggleOff(_SPECULARHIGHLIGHTS_OFF)]_SpecularHighlights ("Specular Highlights", Float) = 1.0
		[ToggleOff(_GLOSSYREFLECTIONS_OFF)]_GlossyReflections ("Glossy Reflections", Float) = 1.0
	}
	SubShader
	{
		Tags
		{
			"RenderType"="TransparentCutout"
			"Queue"="AlphaTest+0"
			"IgnoreProjector"="True"
			"DisableBatching"="True"
		}

		Cull [_Culling]

		CGINCLUDE
			#include "UnityCG.cginc"
			#include "Lighting.cginc"
			#include "AutoLight.cginc"
			#include "UnityPBSLighting.cginc"

			#pragma shader_feature _ _SPECULARHIGHLIGHTS_OFF
			#pragma shader_feature _ _GLOSSYREFLECTIONS_OFF
			#pragma shader_feature _ BLOOM
			#pragma shader_feature _ BLOOM_LOW
			#pragma shader_feature _ FINALPASS
			
			uniform float4 _Color;
			uniform float4 _SpecularColor;
			uniform float _Emission;
			uniform float _Metallic;
			uniform float _Smoothness;
			uniform float _AnisotropyA;
			uniform float _AnisotropyB;
			uniform float _TangentA;
			uniform float _TangentB;
			uniform float _GlossA;
			uniform float _GlossB;
			uniform sampler2D _MainTex;
			uniform sampler2D _BumpMap;
			uniform sampler2D _TangentShiftTex;
			uniform sampler2D _OcclusionMap;
			uniform float4 _MainTex_ST;
			uniform float4 _TangentShiftTex_ST;
			uniform float _Cutoff;
			uniform float _ClampCutoff;
			uniform float _AlphaSharp;
			uniform float _BumpScale;
			uniform float _OcclusionScale;
			uniform float _FakeDirectional;

			// Workaround for ShaderLab issues with DX11 properties. Thanks, Lyuma!
			#if defined(SHADER_STAGE_VERTEX) || defined(SHADER_STAGE_FRAGMENT) || defined(SHADER_STAGE_DOMAIN) || defined(SHADER_STAGE_HULL) || defined(SHADER_STAGE_GEOMETRY)
			#define TEX2DHALF Texture2D<half4>
			#define TEXLOAD(tex, uvcoord) tex.Load(uvcoord)
			#else
			#define precise
			#define centroid
			#define TEX2DHALF float4
			#define TEXLOAD(tex, uvcoord) half4(1,0,1,1)
			#endif

			struct v2f
			{
				#ifndef UNITY_PASS_SHADOWCASTER
				float4 pos : SV_POSITION;
				float3 normal : NORMAL;
				float3 wPos : TEXCOORD0;
				SHADOW_COORDS(3)
				#else
				V2F_SHADOW_CASTER;
				#endif
				float2 uv : TEXCOORD1;
				centroid float3 tangent : TEXCOORD4_centroid;
				centroid float3 bitangent : TEXCOORD5_centroid;
			};

			struct appdata_full_c {
			    float4 vertex : POSITION;
			    centroid float4 tangent : TANGENT_centroid;
			    float3 normal : NORMAL;
			    float4 texcoord : TEXCOORD0;
			    float4 texcoord1 : TEXCOORD1;
			    float4 texcoord2 : TEXCOORD2;
			    float4 texcoord3 : TEXCOORD3;
			    centroid fixed4 color : COLOR_centroid;
			    UNITY_VERTEX_INPUT_INSTANCE_ID
			};


			v2f vert(appdata_full_c v)
			{
				v2f o = (v2f) 0;
				#ifdef UNITY_PASS_SHADOWCASTER
				TRANSFER_SHADOW_CASTER_NOPOS(o, o.pos);
				#else
				o.wPos = mul(unity_ObjectToWorld, v.vertex);
				o.pos = UnityWorldToClipPos(o.wPos);
				o.normal = UnityObjectToWorldNormal(v.normal);
				TRANSFER_SHADOW(o);
				o.tangent = UnityObjectToWorldDir(v.tangent.xyz);
			    half sign = v.tangent.w * unity_WorldTransformParams.w;
				o.bitangent = cross(o.normal, o.tangent) * sign;
				#endif
				o.uv = TRANSFORM_TEX(v.texcoord.xy, _MainTex);
				return o;
			}

//-----------------------------------------------------------------------------
// Helper functions
//-----------------------------------------------------------------------------

// "R2" dithering
float intensity(float2 pixel) {
    const float a1 = 0.75487766624669276;
    const float a2 = 0.569840290998;
    return frac(a1 * float(pixel.x) + a2 * float(pixel.y));
}

// Interleaved Gradient Noise from "NEXT GENERATION POST PROCESSING IN CALL OF DUTY: ADVANCED WARFARE" http://advances.realtimerendering.com/s2014/index.html 
float GradientNoise(float2 pixel)
{
	const float3 magic = float3(0.06711056, 0.00583715, 52.9829189);
	return frac(magic.z * frac(dot(pixel, magic.xy)));
}

float T(float z) {
    return z >= 0.5 ? 2.-2.*z : 2.*z;
}

float3 ShiftTangent (float3 T, float3 N, float shift) 
{
	float3 shiftedT = T + shift * N;
	return normalize(shiftedT);
}

// From "From mobile to high-end PC: Achieving high quality anime style rendering on Unity"
float StrandSpecular(float3 T, float3 V, float3 L, float3 H, float exponent, float strength)
{
	//float3 H = normalize(L+V);
	float dotTH = dot(T, H);
	float sinTH = sqrt(1.0-dotTH*dotTH)+0.001;
	float dirAtten = smoothstep(-1.0, 0.0, dotTH);
	return dirAtten * pow(sinTH, exponent) * strength;
}

struct Interpolators {
	float3 normal;
	float3 tangent;
	float3 bitangent;
};

//-----------------------------------------------------------------------------
// BRDF based on implementation in Filament.
// https://github.com/google/filament
//-----------------------------------------------------------------------------

float D_GGX_Anisotropic(float NoH, float ToH, float BoH,
		float roughness, float gloss) {

	float anisotropy = _AnisotropyA * gloss;
    // The values at and ab are perceptualRoughness^2
	float at = max(roughness * (1.0 + anisotropy), 0.002);
	float ab = max(roughness * (1.0 - anisotropy), 0.002);
    float a2 = at * ab;
    float3 d = float3(ab * ToH, at * BoH, a2 * NoH);
    d += !d * 1e-4f;
    float d2 = dot(d, d);
    float b2 = a2 / d2;
    return a2 * b2 * b2 * UNITY_INV_PI;
}

float V_SmithGGXCorrelated_Anisotropic(float at, float ab, float ToV, float BoV,
        float ToL, float BoL, float NoV, float NoL) {
    // Heitz 2014, "Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs"
    float lambdaV = NoL * length(float3(at * ToV, ab * BoV, NoV));
    float lambdaL = NoV * length(float3(at * ToL, ab * BoL, NoL));
    float v = 0.5f / (lambdaV + lambdaL + 1e-7f);
    return saturate(v);
}

//-----------------------------------------------------------------------------
// Helper functions for roughness
//-----------------------------------------------------------------------------

float RoughnessToPerceptualRoughness(float roughness)
{
    return sqrt(roughness);
}

float RoughnessToPerceptualSmoothness(float roughness)
{
    return 1.0 - sqrt(roughness);
}

float PerceptualSmoothnessToRoughness(float perceptualSmoothness)
{
    return (1.0 - perceptualSmoothness) * (1.0 - perceptualSmoothness);
}

float PerceptualSmoothnessToPerceptualRoughness(float perceptualSmoothness)
{
    return (1.0 - perceptualSmoothness);
}

float PerceptualRoughnessToPerceptualSmoothness(float perceptualRoughness)
{
    return (1.0 - perceptualRoughness);
}

// Return modified perceptualSmoothness based on provided variance (get from GeometricNormalVariance + TextureNormalVariance)
float NormalFiltering(float perceptualSmoothness, float variance, float threshold)
{
    float roughness = PerceptualSmoothnessToRoughness(perceptualSmoothness);
    // Ref: Geometry into Shading - http://graphics.pixar.com/library/BumpRoughness/paper.pdf - equation (3)
    float squaredRoughness = saturate(roughness * roughness + min(2.0 * variance, threshold * threshold)); // threshold can be really low, square the value for easier control

    return RoughnessToPerceptualSmoothness(sqrt(squaredRoughness));
}

// Reference: Error Reduction and Simplification for Shading Anti-Aliasing
// Specular antialiasing for geometry-induced normal (and NDF) variations: Tokuyoshi / Kaplanyan et al.'s method.
// This is the deferred approximation, which works reasonably well so we keep it for forward too for now.
// screenSpaceVariance should be at most 0.5^2 = 0.25, as that corresponds to considering
// a gaussian pixel reconstruction kernel with a standard deviation of 0.5 of a pixel, thus 2 sigma covering the whole pixel.
float GeometricNormalVariance(float3 geometricNormalWS, float screenSpaceVariance)
{
    float3 deltaU = ddx(geometricNormalWS);
    float3 deltaV = ddy(geometricNormalWS);

    return screenSpaceVariance * (dot(deltaU, deltaU) + dot(deltaV, deltaV));
}

// Return modified perceptualSmoothness
float GeometricNormalFiltering(float perceptualSmoothness, float3 geometricNormalWS, float screenSpaceVariance, float threshold)
{
    float variance = GeometricNormalVariance(geometricNormalWS, screenSpaceVariance);
    return NormalFiltering(perceptualSmoothness, variance, threshold);
}


//-----------------------------------------------------------------------------
// BRDF
// Based on Unity's Standard BRDF 1
//-----------------------------------------------------------------------------

float getAnisoD (float NoH, float3 halfDir, float3 tangent, float3 normal, float roughness, float tangentShift) 
{
	half3 b; float ToH; float BoH;

    b = cross(normal + float3(0, 0, _TangentA), tangent);

    ToH = dot(tangent, halfDir);
    BoH = dot(b, halfDir);

	//float3 shiftedTangent1 = ShiftTangent(tangent, normal, 0 + _TangentA);
	float D1 = D_GGX_Anisotropic(NoH, ToH, BoH, roughness, _GlossA);

    b = cross(normal + float3(0, 0, _TangentB), tangent);

    ToH = dot(tangent, halfDir);
    BoH = dot(b, halfDir);

	//float3 shiftedTangent2 = ShiftTangent(tangent, normal, tangentShift + _TangentB);
	float D2 = D_GGX_Anisotropic(NoH, ToH, BoH, roughness, saturate(_GlossB+tangentShift));
	return min(D1 + D2, 100);
}

half4 BRDF_Hair_PBS (half3 diffColor, half3 specColor, half oneMinusReflectivity, half smoothness, half tangentShift,
    Interpolators i, float3 viewDir,
    UnityLight light, UnityIndirect gi)
{
    float perceptualRoughness = SmoothnessToPerceptualRoughness (smoothness);
    float3 halfDir = Unity_SafeNormalize (float3(light.dir) + viewDir);

#define UNITY_HANDLE_CORRECTLY_NEGATIVE_NDOTV_local 0

#if UNITY_HANDLE_CORRECTLY_NEGATIVE_NDOTV_local
    // The amount we shift the normal toward the view vector is defined by the dot product.
    half shiftAmount = dot(i.normal, viewDir);
    i.normal = shiftAmount < 0.0f ? i.normal + viewDir * (-shiftAmount + 1e-5f) : i.normal;
    // A re-normalization should be applied here but as the shift is small we don't do it to save ALU.
    //normal = normalize(normal);

    half nv = saturate(dot(i.normal, viewDir)); // TODO: this saturate should no be necessary here
#else
    half nv = abs(dot(i.normal, viewDir));    // This abs allow to limit artifact
#endif

    // Classic approximation for hair scattering light with biased N.L
    float nl = saturate(lerp(.25, 1.0, dot(i.normal, light.dir)));
    float nh = saturate(dot(i.normal, halfDir));

    half lv = saturate(dot(light.dir, viewDir));
    half lh = saturate(dot(light.dir, halfDir));

    // Diffuse term
    half diffuseTerm = DisneyDiffuse(nv, nl, lh, perceptualRoughness) * nl;

    // Specular term
    float roughness = PerceptualRoughnessToRoughness(perceptualRoughness);

    // GGX with roughness at 0 would mean no specular at all, 
    // max(roughness, 0.002) matches HDRP roughness remapping. 
    roughness = max(roughness, 0.002);

    // More accurate visibility term instead of non-anisotropic?
    // Probably not worth it.
	#if 0
	float TdotL = dot(i.tangent, light.dir);
	float BdotL = dot(i.bitangent, light.dir);
	float TdotV = dot(i.tangent, viewDir);
	float BdotV = dot(i.bitangent, light.dir);

	float V = V_SmithGGXCorrelated_Anisotropic (at, ab, TdotV, BdotV, TdotL, BdotL, nv, nl);
	#else
	float V = SmithJointGGXVisibilityTerm (nl, nv, roughness);
	#endif

    //float D = GGXTerm (nh, roughness); // Original 
    float D = getAnisoD(nh, halfDir, i.tangent, i.normal, roughness, tangentShift);

    float specularTerm = V*D * UNITY_PI; // Torrance-Sparrow model, Fresnel is applied later

#   ifdef UNITY_COLORSPACE_GAMMA
        specularTerm = sqrt(max(1e-4h, specularTerm));
#   endif

    // specularTerm * nl can be NaN on Metal in some cases, use max() to make sure it's a sane value
    // Setting this to zero doesn't work. ???
    specularTerm = max(1e-4h, specularTerm * nl);
#if defined(_SPECULARHIGHLIGHTS_OFF)
    specularTerm = 0.0;
#endif

    // surfaceReduction = Int D(NdotH) * NdotH * Id(NdotL>0) dH = 1/(roughness^2+1)
    half surfaceReduction;
#   ifdef UNITY_COLORSPACE_GAMMA
        surfaceReduction = 1.0-0.28*roughness*perceptualRoughness;      // 1-0.28*x^3 as approximation for (1/(x^4+1))^(1/2.2) on the domain [0;1]
#   else
        surfaceReduction = 1.0 / (roughness*roughness + 1.0);           // fade \in [0.5;1]
#   endif

    // To provide true Lambert lighting, we need to be able to kill specular completely.
    specularTerm *= any(specColor) ? 1.0 : 0.0;

    half grazingTerm = saturate(smoothness + (1-oneMinusReflectivity));
    half3 color =   
    // This is wrong, but it doesn't look too bad.
    // Remove diffuse light if it's guesstimated.
    				diffColor * (gi.diffuse + (light.color * (!any(_WorldSpaceLightPos0.xyz))) * diffuseTerm)
                    + specularTerm * light.color * FresnelTerm (specColor, lh)
                    + surfaceReduction * gi.specular * FresnelLerp (specColor, grazingTerm, nv);
    return half4(color, 1);
}

inline void applyAlphaClip(inout float alpha, float cutoff, float2 pos, bool sharpen)
{
    // Get the amount of MSAA samples present
    #if (SHADER_TARGET > 40)
    half samplecount = GetRenderTargetSampleCount();
    #else
    half samplecount = 1;
    #endif

    pos += (_SinTime.x%4) * 0.7;
    // Switch between dithered alpha and sharp-edge alpha.
        if (!sharpen) {
            alpha = (1+cutoff) * alpha - cutoff;
            alpha = saturate(alpha + alpha*_ClampCutoff);
            float mask = (T(intensity(pos)));
            const float width = 1 / (samplecount*2-1);
            alpha = alpha - (mask * sqrt(1-(alpha)) * width);
        }
        else {
            alpha = ((alpha - cutoff) / max(fwidth(alpha), 0.0001) + 0.5);
        }
    // If 0, remove now.
    clip (alpha - 1.0/255.0);
}

		#ifndef UNITY_PASS_SHADOWCASTER
		float4 frag(v2f i, uint facing : SV_IsFrontFace) : SV_TARGET
		{
    		half3 normalTangent = UnpackScaleNormal(tex2D (_BumpMap, i.uv), _BumpScale);
		    // Thanks, Xiexe!
		    half3 tspace0 = half3(i.tangent.x, i.bitangent.x, i.normal.x);
		    half3 tspace1 = half3(i.tangent.y, i.bitangent.y, i.normal.y);
		    half3 tspace2 = half3(i.tangent.z, i.bitangent.z, i.normal.z);

		    half3 calcedNormal;
		    calcedNormal.x = dot(tspace0, normalTangent);
		    calcedNormal.y = dot(tspace1, normalTangent);
		    calcedNormal.z = dot(tspace2, normalTangent);
		    
		    float3 normal = normalize(calcedNormal);
		    half3 bumpedTangent = (cross(i.bitangent, calcedNormal));
		    half3 bumpedBitangent = (cross(calcedNormal, bumpedTangent));

		    // Flip normals not facing the camera already, but this is bad for hair...
			//normal.z *= facing? 1 : -1; 
			float4 texCol = tex2D(_MainTex, i.uv) * _Color;
			float occlusion = LerpOneTo(tex2D(_OcclusionMap, i.uv).g, _OcclusionScale);

			float alpha = texCol.a;

			applyAlphaClip(alpha, _Cutoff, i.pos.xy, _AlphaSharp);

			float2 uv = i.uv;

			UNITY_LIGHT_ATTENUATION(attenuation, i, i.wPos.xyz);

			float3 specularTint;
			float oneMinusReflectivity;
			float smoothness = _Smoothness;
			smoothness = GeometricNormalFiltering(smoothness, normal, 0.5, 0.25);
			
			#if !defined(BLOOM) // Metalness mode
				oneMinusReflectivity = OneMinusReflectivityFromMetallic(_Metallic);
				float3 albedo = DiffuseAndSpecularFromMetallic(
					texCol, _Metallic, specularTint, oneMinusReflectivity
				);
			#else  // Specular colour mode
				oneMinusReflectivity = 1 - SpecularStrength(_SpecularColor); 
				float3 albedo = texCol;
			#endif

			#if !defined(BLOOM) // Metalness mode
				albedo = EnergyConservationBetweenDiffuseAndSpecular(
				texCol, texCol*_Metallic, oneMinusReflectivity);
				#if defined(FINALPASS) // "Energy convervation"
				specularTint = texCol*_Metallic;
				#else
				specularTint = texCol;
				#endif
			#else  // Specular colour mode
				albedo = EnergyConservationBetweenDiffuseAndSpecular(
				texCol, _SpecularColor*_Metallic, oneMinusReflectivity);
				#if defined(FINALPASS) // "Energy convervation"
				specularTint = _SpecularColor*_Metallic;
				#else
				specularTint = _SpecularColor;
				#endif
			#endif

			float3 viewDir = normalize(_WorldSpaceCameraPos - i.wPos);
			UnityLight light;
			light.color = attenuation * _LightColor0.rgb;
			light.dir = Unity_SafeNormalize(UnityWorldSpaceLightDir(i.wPos));

			// Direction may be wrong here, but there doesn't seem to be a better alternative
			float3 anisotropicT = normalize(UnityObjectToWorldDir(float3(1, 0, 0)));
			float3 anisotropicB = normalize(cross(i.normal, anisotropicT));

			#if !defined(BLOOM_LOW) // Use shift texture
			float tangentShift = dot(0.2 + texCol.rgb - tex2Dlod(_MainTex, float4(i.uv, 0, 7)).rgb , 0.5);
			#else
			float tangentShift = tex2D(_TangentShiftTex, i.uv * _TangentShiftTex_ST.xy + _TangentShiftTex_ST.zw);
			#endif

			UnityIndirect indirectLight;
			#ifdef UNITY_PASS_FORWARDADD
			indirectLight.diffuse = indirectLight.specular = 0;
			#else
			indirectLight.diffuse = max(0, ShadeSH9(float4(normal, 1)));

			float3  anisotropyDirection = _AnisotropyA >= 0.0 ? anisotropicB : anisotropicT;
			float3  anisotropicTangent  = cross(anisotropyDirection, viewDir);
			float3  anisotropicNormal   = cross(anisotropicTangent, anisotropyDirection);
			float bendFactor          = abs(_AnisotropyA) * saturate(5.0 * SmoothnessToPerceptualRoughness(smoothness));
			float3  bentNormal          = normalize(lerp(i.normal, anisotropicNormal, bendFactor));

			float3 reflectionDir = reflect(-viewDir, bentNormal);

			Unity_GlossyEnvironmentData envData;
			envData.roughness = 1 - smoothness;
			envData.reflUVW = reflectionDir;
			    #ifdef _GLOSSYREFLECTIONS_OFF
			        indirectLight.specular = unity_IndirectSpecColor.rgb;
			    #else
				indirectLight.specular = Unity_GlossyEnvironment(
					UNITY_PASS_TEXCUBE(unity_SpecCube0), unity_SpecCube0_HDR, envData
				);
				#endif
			#endif

			indirectLight.specular *= occlusion;

			#ifdef UNITY_PASS_FORWARDBASE
			// Guesstimate a light direction/color if none exists for specular highlights
			if (_FakeDirectional && !any(_WorldSpaceLightPos0.xyz)) {
				// unity_IndirectSpecColor is derived from the skybox, which doesn't always make sense.
				//light.color = indirectLight.diffuse;
				light.dir = Unity_SafeNormalize(light.dir + unity_SHAr.xyz + unity_SHAg.xyz + unity_SHAb.xyz);
    			light.color = ShadeSH9(half4(light.dir, 1.0));
			}
			#endif

			// Workaround an issue with corrected NdotV where backfaces are megaflares.
			light.color *=  (facing? 1 : 0);

			Interpolators iii = (Interpolators)0;
			iii.normal = normal;
			iii.tangent = normalize(i.tangent);
			iii.bitangent = normalize(i.bitangent);
			float3 col = BRDF_Hair_PBS(
				albedo, specularTint,
				oneMinusReflectivity, smoothness, tangentShift,
				iii, viewDir,
				light, indirectLight
			);
			
			//Apply emission
			col.rgb += texCol * _Emission;

			#ifdef UNITY_PASS_FORWARDADD
			return float4(col, 0);
			#else
			return float4(col, alpha);
			#endif
		}
		#else
		float4 frag(v2f i) : SV_Target
		{
			float alpha = _Color.a;
			if (_Color.a > 0)
				alpha *= tex2D(_MainTex, i.uv).a;

			applyAlphaClip(alpha, _Cutoff, i.pos.xy, _AlphaSharp);

			SHADOW_CASTER_FRAGMENT(i)
		}
		#endif
		ENDCG

		Pass
		{
			Tags { "LightMode" = "ForwardBase" }
            AlphaToMask On
			CGPROGRAM
			#pragma vertex vert
			#pragma fragment frag
			#pragma multi_compile_fwdbase_fullshadows
			#pragma multi_compile UNITY_PASS_FORWARDBASE
			ENDCG
		}

		Pass
		{
			Tags { "LightMode" = "ForwardAdd" }
			Blend One One
			CGPROGRAM
			#pragma vertex vert
			#pragma fragment frag
			#pragma multi_compile_fwdadd_fullshadows
			#pragma multi_compile UNITY_PASS_FORWARDADD
			ENDCG
		}

		Pass
		{
			Tags { "LightMode" = "ShadowCaster" }
			CGPROGRAM
			#pragma vertex vert
			#pragma fragment frag
			#pragma multi_compile_shadowcaster
			#pragma multi_compile UNITY_PASS_SHADOWCASTER
			ENDCG
		}
	}
	Fallback "Standard"
}
