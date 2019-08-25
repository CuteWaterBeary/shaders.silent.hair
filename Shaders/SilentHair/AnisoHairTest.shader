Shader "Silent/Hair Anisotropic"
// Thanks to d4rkpl4y3r for providing the base vert/frag Standard lighting shader!
{
	Properties
	{
		[HDR] _Color("Tint", Color) = (1,1,1,1)
		_MainTex("Albedo", 2D) = "white" {}
		_Cutoff("Cutout", Range(0,1)) = .5
		_BumpMap("Normals", 2D) = "bump" {}
		//_SpecularColor("Specular Color", Color) = (0.5, 0.5, 0.5, 1.0)
		[Gamma] _Metallic("Metallic", Range(0, 1)) = 0
		_Smoothness("Smoothness", Range(0, 1)) = 0
		[Gamma]_AnisotropyA("Anisotropy α", Range(-1, 1)) = 0
		[Gamma]_AnisotropyB("Anisotropy β", Range(-1, 1)) = 0
		[Enum(Off, 0, Front, 1, Back, 2)] _Culling ("Culling Mode", Int) = 2
		//[ToggleOff(_SPECULARHIGHLIGHTS_OFF)]_SpecularHighlights ("Specular Highlights", Float) = 1.0
		[ToggleOff(_GLOSSYREFLECTIONS_OFF)]_GlossyReflections ("Glossy Reflections", Float) = 1.0
	}
	SubShader
	{
		Tags
		{
			"RenderType"="TransparentCutout"
			"Queue"="AlphaTest+0"
			"IgnoreProjector"="True"
		}

		Cull [_Culling]

		CGINCLUDE
			#include "UnityCG.cginc"
			#include "Lighting.cginc"
			#include "AutoLight.cginc"
			#include "UnityPBSLighting.cginc"

			uniform float4 _Color;
			//uniform float4 _SpecularColor;
			uniform float _Metallic;
			uniform float _Smoothness;
			uniform float _AnisotropyA;
			uniform float _AnisotropyB;
			uniform sampler2D _MainTex;
			uniform sampler2D _BumpMap;
			uniform float4 _MainTex_ST;
			uniform float _Cutoff;

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
				float3 tangent : TEXCOORD4;
				float3 bitangent : TEXCOORD5;
			};

			struct appdata_full_c {
			    float4 vertex : POSITION;
			    float4 tangent : TANGENT_centroid;
			    float3 normal : NORMAL;
			    float4 texcoord : TEXCOORD0;
			    float4 texcoord1 : TEXCOORD1;
			    float4 texcoord2 : TEXCOORD2;
			    float4 texcoord3 : TEXCOORD3;
			    fixed4 color : COLOR_centroid;
			    UNITY_VERTEX_INPUT_INSTANCE_ID
			};


			v2f vert(appdata_full_c v)
			{
				v2f o;
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

			// Helper functions

			// "R2" dithering
			float intensity(float2 pixel) {
			    const float a1 = 0.75487766624669276;
			    const float a2 = 0.569840290998;
			    return frac(a1 * float(pixel.x) + a2 * float(pixel.y));
			}

			float T(float z) {
			    return z >= 0.5 ? 2.-2.*z : 2.*z;
			}

			// Implementation: https://github.com/CaffeineViking/vkhr
			// Based on "Rendering Hair with Three Dimensional Textures" by J. T. Kajiya and T. L. Kay.
			float kajiya_kay(float p, float3 tangent, float3 light, float3 eye,	inout float specular) {
			    float cosTL = dot(tangent, light);
			    float cosTE = dot(tangent, eye);

			    float cosTL_squared = cosTL*cosTL;
			    float cosTE_squared = cosTE*cosTE;

			    float one_minus_cosTL_squared = 1.0f - cosTL_squared;
			    float one_minus_cosTE_squared = 1.0f - cosTE_squared;

			    float sinTL = sqrt(one_minus_cosTL_squared);
			    float sinTE = sqrt(one_minus_cosTE_squared);

			    specular = pow(max(cosTL*cosTE + sinTL*sinTE, 0), p);

			    return sinTL;
			}

			// BRDF based on implementation in Filament.
			// https://github.com/google/filament

			float D_GGX_Anisotropic(float NoH, const float3 h,
					const float3 t, const float3 b, float at, float ab) {
			    float ToH = dot(t, h);
			    float BoH = dot(b, h);
			    float a2 = at * ab;
			    float3 v = float3(ab * ToH, at * BoH, a2 * NoH);
			    float v2 = dot(v, v);
			    float w2 = a2 / v2;
			    return a2 * w2 * w2 * UNITY_INV_PI;
			}

			float V_SmithGGXCorrelated_Anisotropic(float at, float ab, float ToV, float BoV,
			        float ToL, float BoL, float NoV, float NoL) {
			    // Heitz 2014, "Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs"
			    float lambdaV = NoL * length(float3(at * ToV, ab * BoV, NoV));
			    float lambdaL = NoV * length(float3(at * ToL, ab * BoL, NoL));
			    float v = 0.5 / (lambdaV + lambdaL + 1e-7f);
			    return v;
			}

			// From "From mobile to high-end PC: Achieving high quality anime style rendering on Unity"
			float3 ShiftTangent (float3 T, float3 N, float shift) 
			{
				float3 shiftedT = T + shift * N;
				return normalize(shiftedT);
			}

			float StrandSpecular(float3 T, float3 V, float3 L, float3 H, float exponent, float strength)
			{
				//float3 H = normalize(L+V);
				float dotTH = dot(T, H);
				float sinTH = sqrt(1.0-dotTH*dotTH);
				float dirAtten = smoothstep(-1.0, 0.0, dotTH);
				return dirAtten * pow(sinTH, exponent) * strength;
			}

struct Interpolators {
	float3 normal;
	float3 tangent;
	float3 bitangent;
};

// Based on Unity's Standard BRDF 1
half4 BRDF_Hair_PBS (half3 diffColor, half3 specColor, half oneMinusReflectivity, half smoothness, half tangentShift,
    Interpolators i, float3 viewDir,
    UnityLight light, UnityIndirect gi)
{
    float perceptualRoughness = SmoothnessToPerceptualRoughness (smoothness);
    float3 halfDir = Unity_SafeNormalize (float3(light.dir) + viewDir);

    half nv = abs(dot(i.normal, viewDir));    // This abs allow to limit artifact

    float nl = saturate(dot(i.normal, light.dir));
    float nh = saturate(dot(i.normal, halfDir));

    half lv = saturate(dot(light.dir, viewDir));
    half lh = saturate(dot(light.dir, halfDir));

    // Diffuse term
    half diffuseTerm = DisneyDiffuse(nv, nl, lh, perceptualRoughness) * nl;

    // Specular term
    float roughness = PerceptualRoughnessToRoughness(perceptualRoughness);

	float at = max(roughness * (1.0 + _AnisotropyA), 0.001);
	float ab = max(roughness * (1.0 - _AnisotropyB), 0.001);

    // GGX with roughtness to 0 would mean no specular at all, using max(roughness, 0.002) here to match HDrenderloop roughtness remapping.
    roughness = max(roughness, 0.002);
	#if 1
	float TdotL = dot(i.tangent, light.dir);
	float BdotL = dot(i.bitangent, light.dir);
	float TdotV = dot(i.tangent, viewDir);
	float BdotV = dot(i.bitangent, light.dir);

	// Accurate but probably expensive
	float V = V_SmithGGXCorrelated_Anisotropic (at, ab, TdotV, BdotV, TdotL, BdotL, nv, nl);
	#else
	float V = SmithJointGGXVisibilityTerm (nl, nv, roughness);
	#endif
    //float D = GGXTerm (nh, roughness); // Original 
	float3 shiftedTangent = ShiftTangent(i.tangent, i.normal, tangentShift);
	float D = D_GGX_Anisotropic(nh, halfDir, shiftedTangent, i.bitangent, at, ab);

	float specK;
	#ifdef UNITY_PASS_FORWARDBASE
	// Guesstimate a light direction if none exists.
	light.color = any(_WorldSpaceLightPos0.xyz) ? light.color : gi.diffuse;
	light.dir = Unity_SafeNormalize(light.dir + unity_SHAr.xyz + unity_SHAg.xyz + unity_SHAb.xyz);
	#endif
	float diffK = kajiya_kay(80.0, shiftedTangent, light.dir, i.normal, specK);

	D += max(specK,0.0001);
	D = min(D, 2); // Hack to fix sparks. 
	diffuseTerm = max(diffK,0.0001); // Not allowed to be zero?

    float specularTerm = V*D * UNITY_PI; // Torrance-Sparrow model, Fresnel is applied later

#   ifdef UNITY_COLORSPACE_GAMMA
        specularTerm = sqrt(max(1e-4h, specularTerm));
#   endif

    // specularTerm * nl can be NaN on Metal in some cases, use max() to make sure it's a sane value
    specularTerm = max(0, specularTerm * nl);
#if defined(_SPECULARHIGHLIGHTS_OFF)
    specularTerm = 0.0;
#endif

	// hair_color, light_bulb_color, hair_exponent,
    // fs_in.tangent, light_direction, eye_normal

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
    half3 color =   //diffColor * (gi.diffuse + light.color * diffuseTerm)
    // This is wrong, but it doesn't look too bad.
    				diffuseTerm;
    return half4(color, 1);
}

			#ifndef UNITY_PASS_SHADOWCASTER
			float4 frag(v2f i) : SV_TARGET
			{
				fixed3 normalTangent = UnpackNormal( tex2D (_BumpMap, i.uv));
				float3 normal = normalize(i.tangent * normalTangent.x + i.bitangent * normalTangent.y + i.normal * normalTangent.z); 
				float4 texCol = tex2D(_MainTex, i.uv) * _Color;

				float alpha = texCol.a;
				float mask = (T(intensity(i.pos.xy + _SinTime.x%4)));
				alpha = saturate(alpha + alpha * mask); 
				clip(alpha - 1.0/255.0); // Attempt to fix no-HDR bug

				float2 uv = i.uv;

				UNITY_LIGHT_ATTENUATION(attenuation, i, i.wPos.xyz);

				float3 specularTint;
				float oneMinusReflectivity;
				float smoothness = _Smoothness;
				
				//float3 albedo = DiffuseAndSpecularFromMetallic(
				//	texCol, _Metallic, specularTint, oneMinusReflectivity
				//);
				//
				//// DESTROY energy conservation
				//albedo = specularTint = texCol;

				float3 albedo = EnergyConservationBetweenDiffuseAndSpecular(
					texCol, texCol*_Metallic, oneMinusReflectivity);
				specularTint = texCol;

				float3 viewDir = normalize(_WorldSpaceCameraPos - i.wPos);
				UnityLight light;
				light.color = attenuation * _LightColor0.rgb;
				light.dir = normalize(UnityWorldSpaceLightDir(i.wPos));
				UnityIndirect indirectLight;
				#ifdef UNITY_PASS_FORWARDADD
				indirectLight.diffuse = indirectLight.specular = 0;
				#else
				indirectLight.diffuse = max(0, ShadeSH9(float4(normal, 1)));
				float3 reflectionDir = reflect(-viewDir, normal);
				Unity_GlossyEnvironmentData envData;
				envData.roughness = 1 - smoothness;
				envData.reflUVW = reflectionDir;
				indirectLight.specular = Unity_GlossyEnvironment(
					UNITY_PASS_TEXCUBE(unity_SpecCube0), unity_SpecCube0_HDR, envData
				);
				#endif

				Interpolators iii = (Interpolators)0;
				iii.normal = normal;
				iii.tangent = i.tangent;
				iii.bitangent = i.bitangent;

				float tangentShift = dot(texCol - tex2Dlod(_MainTex, float4(i.uv, 0, 7)) , 1.0);
				float3 col = BRDF_Hair_PBS(
					albedo, specularTint,
					oneMinusReflectivity, smoothness, tangentShift,
					iii, viewDir,
					light, indirectLight
				);

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
				if (_Cutoff > 0)
					alpha *= tex2D(_MainTex, i.uv).a;

				float mask = (T(intensity(i.pos.xy + _SinTime.x%4)));
				//alpha *= alpha;
				alpha = saturate(alpha + alpha * mask); 
				clip(alpha - _Cutoff);
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
