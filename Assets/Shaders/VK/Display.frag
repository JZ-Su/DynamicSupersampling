/******************************************************************************
This file is part of the Newcastle Vulkan Tutorial Series

Author:Rich Davison
Contact:richgdavison@gmail.com
License: MIT (see LICENSE file at the top of the source tree)
*//////////////////////////////////////////////////////////////////////////////

#version 450
#extension GL_ARB_separate_shader_objects  : enable
#extension GL_ARB_shading_language_420pack : enable

layout (binding  = 0, set = 0) uniform  sampler2D tex;

layout (location = 0) in  vec2 texcoord;
layout (location = 0) out vec4 fragColor;

void main() {
   fragColor 	= texture(tex, texcoord);

   // vec2 pos = texcoord;
   // vec4 centerColor = texture(tex, texcoord);

   // float sigmaSpace = 1.0f;
   // float sigmaColor = 1.0f;
   // float radius = 2.0 * sigmaSpace;
   // float sumWeights = 0.0;
   // vec4 sumColor = vec4(0.0);

   // for (int x = -int(radius); x <= int(radius); x++) {
   //    for (int y = -int(radius); y <= int(radius); y++) {
   //       vec2 neighborPos = pos + vec2(x, y);
   //       vec4 neighborColor = texture(tex, neighborPos);

   //       float spatialWeight = exp(-float(x * x + y * y) / (2.0 * sigmaSpace * sigmaSpace));
   //       float colorWeight = exp(-dot(neighborColor - centerColor, neighborColor - centerColor) / (2.0 * sigmaColor * sigmaColor));

   //       float weight = spatialWeight * colorWeight;
   //       sumColor += neighborColor * weight;
   //       sumWeights += weight;
   //   }
   // }

   // fragColor = sumColor / sumWeights;
}