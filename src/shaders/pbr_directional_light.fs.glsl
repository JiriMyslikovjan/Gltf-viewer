#version 330

in vec3 vViewSpacePosition;
in vec3 vViewSpaceNormal;
in vec2 vTexCoords;

uniform vec3 uLightDirection;
uniform vec3 uLightIntensity;

uniform float uMetallicFactor;
uniform float uRoughnessFactor;
uniform float uOcclusionStrength;
uniform vec3 uEmissiveFactor;

uniform sampler2D uBaseColorTexture;
uniform sampler2D uMetallicRoughnessTexture;
uniform sampler2D uEmissiveTexture;
uniform sampler2D uOcclusionTexture;

uniform vec4 uBaseColorFactor;

uniform int uApplyOcclusion;

out vec3 fColor;

// Constants
const float GAMMA = 2.2;
const float INV_GAMMA = 1. / GAMMA;
const float M_PI = 3.141592653589793;
const float M_1_PI = 1.0 / M_PI;

// We need some simple tone mapping functions
// Basic gamma = 2.2 implementation
// Stolen here:
// https://github.com/KhronosGroup/glTF-Sample-Viewer/blob/master/src/shaders/tonemapping.glsl

// linear to sRGB approximation
vec3 LINEARtoSRGB(vec3 color) { return pow(color, vec3(INV_GAMMA)); }

// sRGB to linear approximation
vec4 SRGBtoLINEAR(vec4 srgbIn)
{
  return vec4(pow(srgbIn.xyz, vec3(GAMMA)), srgbIn.w);
}

void main()
{
  vec3 N = normalize(vViewSpaceNormal);
  vec3 L = uLightDirection;
  vec3 V = normalize(-vViewSpacePosition);
  vec3 H = normalize(L + V);

  vec4 roughnessTexture = texture(uMetallicRoughnessTexture, vTexCoords);
  
  float roughness = uRoughnessFactor * roughnessTexture.g ;
  vec3 metallic = vec3(roughnessTexture.b * uMetallicFactor);

  vec3 dielectricSpecular = vec3(0.04); 
  vec3 black = vec3(0.);

  float VdotH = clamp(dot(V, H), 0., 1.);
  float NdotL = clamp(dot(N, L), 0., 1.);
  float NdotH = clamp(dot(N, H), 0., 1.);
  float NdotV = clamp(dot(N, V), 0., 1.);

  vec4 baseColor = SRGBtoLINEAR(texture(uBaseColorTexture, vTexCoords)) * uBaseColorFactor;

  vec3 c_diff = mix(baseColor.rgb * (1 - dielectricSpecular.r), black ,metallic);
  vec3 f0 = mix(vec3(dielectricSpecular), baseColor.rgb, metallic);
  float alpha = roughness * roughness;

  float baseShlickFactor = 1 - VdotH;
  float shlickFactor = baseShlickFactor * baseShlickFactor;
  shlickFactor *= shlickFactor;
  shlickFactor *= baseShlickFactor;

  float DBaseDenominator = (NdotH * NdotH * (alpha * alpha - 1.) + 1.);
  float D = M_1_PI * (alpha * alpha) / (DBaseDenominator * DBaseDenominator);

  float VisDenominator = NdotL * sqrt((alpha * alpha) + (1 - alpha * alpha) * (NdotV * NdotV)) + 
                       NdotV * sqrt((alpha * alpha) + (1 - alpha * alpha) * (NdotL * NdotL));
  float Vis = VisDenominator > 0. ? 0.5 / VisDenominator : 0.0;

  vec3 F = f0 + (vec3(1) - f0) * shlickFactor;

  vec3 f_diffuse = (1 - F) * (M_1_PI) * c_diff;
  vec3 f_specular = F * Vis * D;

  vec3 emission = SRGBtoLINEAR(texture2D(uEmissiveTexture, vTexCoords)).rgb * uEmissiveFactor;
  
  vec3 color = ((f_diffuse + f_specular) * uLightIntensity * NdotL) + emission;

  if(uApplyOcclusion == 1)
  {
    float ambientOcclusion = texture2D(uOcclusionTexture, vTexCoords).r;
    color = mix(color, color * ambientOcclusion, uOcclusionStrength); 
  }

  fColor = LINEARtoSRGB(color);
}