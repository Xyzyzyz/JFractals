#if defined(UNITY_PROCEDURAL_INSTANCING_ENABLED)
	StructuredBuffer<float3> _Positions;
#endif

float _Step;

void ConfigureProcedural () {
	#if defined(UNITY_PROCEDURAL_INSTANCING_ENABLED)
		float3 position = _Positions[unity_InstanceID];

		unity_ObjectToWorld = 0.0; // Set all elements of the instance's 4x4 transformation matrix to 0
		unity_ObjectToWorld._m03_m13_m23_m33 = float4(position, 1.0); // Modify fourth column with position values and 1 (the last element of the matrix is always 1)
		unity_ObjectToWorld._m00_m11_m22 = _Step; // Diagonal of transformation matrix = scale. Set diagonal to match the step values and correctly scale point coordinates
	#endif
}

void ShaderGraphFunction_float (float3 In, out float3 Out) {
	Out = In;
}

void ShaderGraphFunction_half (half3 In, out half3 Out) {
	Out = In;
}