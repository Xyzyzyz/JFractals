
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;

using static Unity.Mathematics.math;
using quaternion = Unity.Mathematics.quaternion;
using Random = UnityEngine.Random;

class Fractal : MonoBehaviour {

	[BurstCompile(FloatPrecision.Standard, FloatMode.Fast, CompileSynchronously = true)]
	struct UpdateFractalLevelJob : IJobFor { // I = Interface, JobFor = Job that is used for functionality that runs inside for loops
		
		public float spinAngleDelta;
		public float scale;
		public float deltaTime;

		[ReadOnly]
		public NativeArray<FractalPart> parents;
		public NativeArray<FractalPart> parts;

		[WriteOnly]
		public NativeArray<float3x4> matrices;

		public void Execute (int i) {
			FractalPart parent = parents[i / 5];
			FractalPart part = parts[i];
			part.spinAngle += part.spinVelocity * deltaTime;

			float3 upAxis = mul(mul(parent.worldRotation, part.rotation), up()); // Add the part's local rotation + its parent's world orientation to the global up
			float3 sagAxis = cross(up(), upAxis); // cross method returns a vector perpendicular to both arguments
			
			float sagMagnitude = length(sagAxis);
			quaternion baseRotation;
			if (sagMagnitude > 0f) {
				sagAxis /= sagMagnitude; // normalize; cross method does not return a unit vector (depending on arguments) -- vector / its magnitude = unit length vector!
				quaternion sagRotation = quaternion.AxisAngle(sagAxis, part.maxSagAngle * sagMagnitude); // AxisAngle returns a quaternion with rotation {1} around axis {0} -- in this case, 45 degrees around the sag axis
				baseRotation = mul(sagRotation, parent.worldRotation); // Add sag to the parent's world rotation, i.e. the non-sagged starting point
			}
			else {
				baseRotation = parent.worldRotation; // If the part is pointing straight up (i.e. the difference between its up and global up is 0), the length of the sag axis will be 0. 
													 // No good for normalization! This defaults to the parent's rotation (i.e. no sag), which makes sense!
			}

			part.worldRotation = mul(baseRotation, // used to be parent's world rotation 
				mul(part.rotation, quaternion.RotateY(part.spinAngle))
			); // "Adding" rotations via quaternion multiplication -- Second quat is applied first, first quat is applied second -- Order matters!
			part.worldPosition = 
				parent.worldPosition + 
				mul(part.worldRotation, float3(0f, 1.5f * scale, 0f)); // Quaternion-vector multiplication! Rotation of the parent part affects the position of the child part -- duh!
				// On parent -> move in direction multiplied by scale of part (scale is uniform so just using the x component is fine)

			parts[i] = part;
			float3x3 r = float3x3(part.worldRotation) * scale; // 3x3 matrix for rotation and scale
			matrices[i] = float3x4(r.c0, r.c1, r.c2, part.worldPosition); // Transformation matrix -- rotation and scale make up the first three columns and are followed by the position
		}
	}

	[SerializeField, Range(3, 8)] // We need to perform a -2 operation to buffer indices later on, so need 3 in order to avoid dividing by 0
	int depth = 4;

	[SerializeField]
	Gradient gradientA, gradientB;

	[SerializeField]
	Color leafColorA, leafColorB;

	[SerializeField, Range(0f, 90f)]
	float maxSagAngleA = 15f, maxSagAngleB = 25f;

	[SerializeField, Range(0f, 90f)]
	float spinSpeedA = 20f, spinSpeedB = 25f;

	[SerializeField, Range(0f, 1f)]
	float reverseSpinChance = 0.25f;

	[SerializeField]
	Mesh mesh, leafMesh;

	[SerializeField]
	Material material;

	static quaternion[] rotations = {
		quaternion.identity,
		quaternion.RotateZ(-0.5f * PI), quaternion.RotateZ(0.5f * PI),
		quaternion.RotateX(0.5f * PI), quaternion.RotateX(-0.5f * PI)
	};

	NativeArray<FractalPart>[] parts;

	NativeArray<float3x4>[] matrices;

	struct FractalPart {
		public float3 worldPosition;
		public quaternion rotation, worldRotation;
		public float maxSagAngle, spinAngle, spinVelocity;
	}

	ComputeBuffer[] matricesBuffers;
	static readonly int 
		colorAId = Shader.PropertyToID("_ColorA"),
		colorBId = Shader.PropertyToID("_ColorB"),
		matricesId = Shader.PropertyToID("_Matrices"),
		sequenceNumbersId = Shader.PropertyToID("_SequenceNumbers");
	static MaterialPropertyBlock propertyBlock;

	Vector4[] sequenceNumbers;

	void OnEnable () {
		parts = new NativeArray<FractalPart>[depth];
		matrices = new NativeArray<float3x4>[depth];
		matricesBuffers = new ComputeBuffer[depth];

		sequenceNumbers = new Vector4[depth];
		int stride = 12 * 4; // Transformation matrices are 4x4 matrices of float values -- 16 x 4 bytes per float
							 // ...Or they would be if the bottom row (0, 0, 0, 1) wasn't always the same. We can make them 3x4, which means 12 x 4 bytes per float 

		for (int i = 0, length = 1; i < parts.Length; i++, length *= 5) {
			parts[i] = new NativeArray<FractalPart>(length, Allocator.Persistent); // Arguments for NativeArray: Length, How long the array will exist
			matrices[i] = new NativeArray<float3x4>(length, Allocator.Persistent); // Allocator.Persisent because we use the same arrays for every frame
			matricesBuffers[i] = new ComputeBuffer(length, stride);
			sequenceNumbers[i] = new Vector4(Random.value, Random.value, Random.value, Random.value);
		}

		parts[0][0] = CreatePart(0);
		for (int li = 1; li < parts.Length; li++) {
			NativeArray<FractalPart> levelParts = parts[li];
			for (int fpi = 0; fpi < levelParts.Length; fpi += 5) {
				for (int ci = 0; ci < 5; ci++) {
					parts[li][fpi + ci] = CreatePart(ci);
				}
			}
		}

		propertyBlock ??= new MaterialPropertyBlock(); // Null-coalescing argument! var ??= val is equivalent to "If var is null, set var to val" -- if propertyBlock is null, set it to a new materialpropertyblock!
	}

	void OnDisable () {
		for (int i = 0; i < matricesBuffers.Length; i++) {
			matricesBuffers[i].Release();
			parts[i].Dispose();
			matrices[i].Dispose();
		}
		parts = null;
		matrices = null;
		matricesBuffers = null;
		sequenceNumbers = null;
	}

	void OnValidate () { // Called whenever a change is made via the inspector or undo/redo action. EZ switching!
		if (parts != null && enabled) { // Only reset if we're in play mode and not currently disabled
			OnDisable();
			OnEnable();
		}
	}

	void Update () {
		float deltaTime = Time.deltaTime;
		//spinAngleDelta = 0f;
		
		FractalPart rootPart = parts[0][0];
		rootPart.spinAngle += rootPart.spinVelocity * deltaTime; // Remember quat "multiplication" -- root part rotation * delta rotation = root part rotation applied on top of delta rotation
		rootPart.worldRotation =
			mul(transform.rotation, mul(rootPart.rotation, quaternion.RotateY(rootPart.spinAngle)) // Apply main object's rotation
			); // Making a new quaternion every update to avoid floating-point errors with quat multiplication
		rootPart.worldPosition = transform.position;
		parts[0][0] = rootPart;

		float objectScale = transform.lossyScale.x;
		float3x3 r = float3x3(rootPart.worldRotation) * objectScale;
		matrices[0][0] = float3x4(r.c0, r.c1, r.c2, rootPart.worldPosition);

		float scale = objectScale; // Apply main object's scale

		JobHandle jobHandle = default;
		for (int li = 1; li < parts.Length; li++) {
			scale *= 0.5f;

			jobHandle = new UpdateFractalLevelJob { // Make a job for parts on the current level -- These fields will be the same for all parts on the current level
				deltaTime = deltaTime,
				scale = scale,
				parents = parts[li - 1],
				parts = parts[li],
				matrices = matrices[li]
			}.ScheduleParallel(parts[li].Length, 5, jobHandle); // Similar to a "for" loop -- First argument is number of iterations, second is struct value used to enforce sequential dependency. 
																   // "default" enforces no constraints
																   // Schedule function returns JobHandle struct that gives status of that job
																   // By passing each jobhandle to the next job, we wait for all jobs to be scheduled before completing any job
			// Execute the job passing index of the part within the level as an argument
		}
		jobHandle.Complete(); // Complete the current JobHandle - the last JobHandle in the sequence. This triggers all previous JobHandles due to the sequential dependancy we built up while scheduling.

		Bounds bounds = new Bounds(rootPart.worldPosition, 3f * objectScale * Vector3.one); // Bound to area around main object
		int leafIndex = matricesBuffers.Length - 1; // Index of the last layer, i.e. the layer at which we will draw the leaves
		for (int i = 0; i < matricesBuffers.Length; i++) {
			ComputeBuffer buffer = matricesBuffers[i];

			buffer.SetData(matrices[i]); // Upload matrix to the GPU

			Color colorA, colorB;
			Mesh instanceMesh;

			if (i == leafIndex) {
				colorA = leafColorA;
				colorB = leafColorB;
				instanceMesh = leafMesh;
			}
			else {
				float gradientInterpolator = i / (matricesBuffers.Length - 2f); // Gradient ends one step before the end, so we have to subtract two
				colorA = gradientA.Evaluate(gradientInterpolator);
				colorB = gradientB.Evaluate(gradientInterpolator);
				instanceMesh = mesh;
			}
			
			propertyBlock.SetColor(colorAId, colorA);
			propertyBlock.SetColor(colorBId, colorB);

			propertyBlock.SetVector(sequenceNumbersId, sequenceNumbers[i]);
			propertyBlock.SetBuffer(matricesId, buffer);
			Graphics.DrawMeshInstancedProcedural(
				instanceMesh, 0, material, bounds, buffer.count, propertyBlock // New argument! Use the property block data to procedurally draw meshes
			);
		}
	}

	FractalPart CreatePart (int childIndex) => new FractalPart {
		maxSagAngle = radians(Random.Range(maxSagAngleA, maxSagAngleB)),
		spinVelocity = (Random.value < reverseSpinChance ? -1f : 1f) * radians(Random.Range(spinSpeedA, spinSpeedB)),
		rotation = rotations[childIndex]
	};
}