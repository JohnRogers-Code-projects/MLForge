# ADR-001: Use ONNX Runtime for ML Inference

## Status

Accepted

## Context

ModelForge needs to serve machine learning models for inference. We need to support models from various ML frameworks (PyTorch, TensorFlow, scikit-learn) while maintaining consistent inference performance and a simple deployment story.

Options considered:

1. **Native framework serving**: Run models in their original frameworks
2. **TensorFlow Serving**: Google's model serving solution
3. **TorchServe**: PyTorch's model serving solution
4. **ONNX Runtime**: Microsoft's cross-platform inference engine
5. **Triton Inference Server**: NVIDIA's inference serving platform

## Decision

We will use **ONNX Runtime** as our inference engine, requiring models to be converted to ONNX format before upload.

### Rationale

1. **Framework agnostic**: ONNX is supported by all major ML frameworks
2. **Performance**: ONNX Runtime provides optimized inference across CPU and GPU
3. **Simplicity**: Single runtime to maintain, regardless of original framework
4. **Portability**: ONNX models are self-contained and portable
5. **Validation**: ONNX format allows schema extraction and input validation
6. **Size**: Lighter weight than full framework runtimes

### Trade-offs

- Users must convert models to ONNX format (one-time cost)
- Some advanced framework features may not convert perfectly
- Dynamic shapes require careful handling

## Consequences

### Positive

- Consistent inference behavior across all models
- Simplified deployment (single runtime dependency)
- Automatic input/output schema extraction
- Good performance without GPU (CPU optimizations)
- Clear model versioning via ONNX opset versions

### Negative

- Model conversion step required for users
- Some models may need adjustment for ONNX compatibility
- Limited to operations supported by ONNX spec

### Mitigations

- Provide documentation on model conversion
- Validate models on upload to catch conversion issues early
- Support multiple ONNX opset versions for compatibility
