
using UnityEngine;

public class GPUGraph : MonoBehaviour {

	// Compute Shader setup
	[SerializeField]
	ComputeShader computeShader;

	static readonly int 
		positionsId = Shader.PropertyToID("_Positions"),
		resolutionId = Shader.PropertyToID("_Resolution"),
		stepId = Shader.PropertyToID("_Step"),
		timeId = Shader.PropertyToID("_Time"),
		transitionProgressId = Shader.PropertyToID("_TransitionProgress");

	[SerializeField]
	Material material;

	[SerializeField]
	Mesh mesh;

	// Visualization settings
	const int maxResolution = 1000;

	[SerializeField, Range(10, maxResolution)]
	int resolution = 10;

	[SerializeField]
	FunctionLibrary.FunctionName function;

	public enum TransitionMode { Cycle, Random }

	[SerializeField]
	TransitionMode transitionMode;

	[SerializeField, Min(0f)]
	float functionDuration = 1f, transitionDuration = 1f;

	float duration;
	bool transitioning;

	FunctionLibrary.FunctionName transitionFunction;

	ComputeBuffer positionsBuffer;

	void OnEnable () {
		positionsBuffer = new ComputeBuffer(maxResolution * maxResolution, 3 * 4); // Always create buffer with max resolution in mind. 
																				   // This means that every point in the graph will be updated regardless of current resolution
																				   // And thus supports runtime changes to resolution without "leaving behind" any points from greater resolutions than the current
	}

	void OnDisable () {
		positionsBuffer.Release();
		positionsBuffer = null;
	}

	void Update () {
		duration += Time.deltaTime;

		// Checks for changes to transition state and current functions
		if (transitioning) {
			if (duration >= transitionDuration) {
				duration -= transitionDuration;
				transitioning = false;
			}
		}
		else if (duration >= functionDuration) {
			duration -= functionDuration;
			transitioning = true;
			transitionFunction = function;
			PickNextFunction();
		}

		UpdateFunctionOnGPU();
	}

	void PickNextFunction () {
		function = transitionMode == TransitionMode.Cycle ? 
			FunctionLibrary.GetNextFunctionName(function) : 
			FunctionLibrary.GetRandomFunctionNameOtherThan(function);
	}

	void UpdateFunctionOnGPU () {
		float step = 2f / resolution;
		int kernelIndex = 
			(int)function + 
			(int)(transitioning ? transitionFunction : function) 
			* FunctionLibrary.FunctionCount; // Add 5x the next function if transitioning, 5x the current function if not
		
		computeShader.SetInt(resolutionId, resolution); // Set resolution in compute shader
		computeShader.SetFloat(stepId, step); // Set step in compute shader
		computeShader.SetFloat(timeId, Time.time); // Set time in compute shader

		if (transitioning) {
			computeShader.SetFloat(transitionProgressId, Mathf.SmoothStep(0, 1, duration / transitionDuration));
		}

		computeShader.SetBuffer(kernelIndex, positionsId, positionsBuffer); // Link position buffer to compute shader kernel. Use FindKernel method when there are multiple kernels!

		int groups = Mathf.CeilToInt(resolution / 8f); // No. of threads in compute shader is 8, 8, 1, so we need to run resolution / 8 groups for first 2 fields. ceiltoint so we don't underestimate.
		computeShader.Dispatch(kernelIndex, groups, groups, 1); // Dispatch compute shader -- arguments: kernel index, number of groups to run (group = thread count in compute shader) x3

		material.SetBuffer(positionsId, positionsBuffer); // For rendering transformation matrices -- check "Point Surface GPU.shader"!
		material.SetFloat(stepId, step); // For rendering transformation matrices -- check "Point Surface GPU.shader"!

		Bounds bounds = new Bounds(Vector3.zero, Vector3.one * (2f + 2f / resolution));
		Graphics.DrawMeshInstancedProcedural(
			mesh, 0, material, bounds, resolution * resolution // second argument is mesh subset, which doesn't matter since we're only using one material
		);
	}
}