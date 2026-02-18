# Diagram-to-Code Converter

A local web application that converts deep learning model diagrams into executable **PyTorch** or **Keras** code.

Accepts **Mermaid flowcharts** or **Draw.io XML**, parses them into an intermediate representation (IR), validates the graph, generates production-quality model code, and optionally polishes output with a local LLM via Ollama.

---

## Features

- **Dual input format** — Mermaid flowchart syntax and Draw.io XML (auto-detected).
- **Dual framework output** — generates valid PyTorch `nn.Module` or Keras `Sequential` models.
- **Deterministic pipeline** — the LLM is never in the critical path; all parsing, validation, and code generation is rule-based.
- **Shape inference** — automatically computes `in_features`, `in_channels`, and flattened dimensions so generated models actually run.
- **Three-pass IR validation** — structural (duplicate IDs, dangling edges), topological (cycles, reachability), and semantic (parameter types and required fields).
- **Generated code validation** — syntax check, import check, and runtime forward pass with a dummy tensor.
- **Optional LLM polishing** — uses Ollama (local) for adding comments and formatting; falls back gracefully if unavailable.
- **Web UI** — drag-and-drop upload, framework selector, live code viewer, and download button.
- **Comprehensive test suite** — 120+ tests covering every module independently.

---

## Supported Layers

| Layer | IR Type | PyTorch | Keras |
|---|---|---|---|
| Input | `Input` | *(function arg)* | `layers.Input` |
| Fully Connected | `Linear` / `Dense` | `nn.Linear` | `layers.Dense` |
| 2D Convolution | `Conv2D` | `nn.Conv2d` | `layers.Conv2D` |
| Max Pooling | `MaxPool2D` | `nn.MaxPool2d` | `layers.MaxPool2D` |
| Flatten | `Flatten` | `nn.Flatten` | `layers.Flatten` |
| Dropout | `Dropout` | `nn.Dropout` | `layers.Dropout` |
| Batch Normalization | `BatchNorm` | `nn.BatchNorm1d/2d` | `layers.BatchNormalization` |
| ReLU | `ReLU` | `nn.ReLU` | `layers.Activation('relu')` |
| Sigmoid | `Sigmoid` | `nn.Sigmoid` | `layers.Activation('sigmoid')` |
| Softmax | `Softmax` | `nn.Softmax` | `layers.Softmax` |

---

## Project Structure

```
app/
├── main.py                  # FastAPI entry point
├── logging_config.py        # Structured logging setup
├── api/
│   └── routes.py            # REST endpoints (/health, /upload, /parse, /generate, /download)
├── parser/
│   ├── base.py              # Abstract parser interface
│   ├── mermaid_parser.py    # Mermaid flowchart → IR
│   ├── drawio_parser.py     # Draw.io XML → IR
│   ├── factory.py           # Auto-detection + parser dispatch
│   └── exceptions.py        # Parser error types
├── ir/
│   └── models.py            # IRNode, IREdge, IRGraph (Pydantic)
├── generator/
│   ├── topo_sort.py         # Topological sort (Kahn's algorithm)
│   ├── shape_tracker.py     # Tensor shape inference engine
│   ├── pytorch_gen.py       # IR → PyTorch nn.Module
│   └── keras_gen.py         # IR → Keras Sequential
├── validator/
│   ├── ir_validator.py      # Three-pass IR graph validation
│   ├── code_validator.py    # Syntax / import / runtime code validation
│   └── exceptions.py        # Validation error types
├── llm/
│   └── ollama_client.py     # Optional Ollama HTTP client
├── tests/
│   ├── test_health.py       # Phase 1: health endpoint
│   ├── test_parser.py       # Phase 2: diagram parsing
│   ├── test_validator.py    # Phase 3: IR validation
│   ├── test_pytorch_gen.py  # Phase 4: PyTorch generation
│   ├── test_keras_gen.py    # Phase 5: Keras generation
│   ├── test_ollama.py       # Phase 6: LLM integration
│   ├── test_api_pipeline.py # Phase 7: API endpoints
│   ├── test_code_validator.py # Phase 8: code validation
│   └── test_frontend.py     # Phase 9: frontend serving
└── frontend/
    └── index.html           # Single-page web UI
```

---

## Quick Start

### 1. Clone and set up

```bash
cd Project_2
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Install ML frameworks

```bash
# PyTorch (CPU)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# TensorFlow (CPU)
pip install tensorflow-cpu
```

### 3. Run the server

```bash
uvicorn app.main:app --reload
```

Open http://localhost:8000 in your browser.

### 4. Run the tests

```bash
python -m pytest app/tests/ -v
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/upload` | Upload a diagram file (multipart form) |
| `POST` | `/parse` | Parse diagram text (JSON body) |
| `POST` | `/generate` | Generate model code from a session |
| `GET` | `/download` | Download generated code |
| `GET` | `/docs` | Interactive Swagger UI |

### Example: full pipeline via curl

```bash
# 1. Parse a Mermaid diagram
curl -X POST http://localhost:8000/parse \
  -H "Content-Type: application/json" \
  -d '{
    "diagram": "graph TD\n    A[Input: shape=784] --> B[Linear: 256]\n    B --> C[ReLU]\n    C --> D[Linear: 10]\n    D --> E[Softmax]"
  }'
# Returns: { "session_id": "abc123...", "ir": {...}, ... }

# 2. Generate PyTorch code
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"session_id": "abc123...", "framework": "pytorch"}'
# Returns: { "code": "import torch\n...", ... }

# 3. Download
curl "http://localhost:8000/download?session_id=abc123..."
```

---

## Diagram Format

### Mermaid

```
graph TD
    A[Input: shape=784] --> B[Linear: 256]
    B --> C[ReLU]
    C --> D[Dropout: 0.5]
    D --> E[Linear: 10]
    E --> F[Softmax]
```

Node label convention:
- `NodeId[Type: key=value, key2=value2]` — explicit params
- `NodeId[Type: value]` — shorthand (primary param)
- `NodeId[Type]` — no params

### Draw.io XML

```xml
<mxGraphModel>
  <root>
    <mxCell id="0"/>
    <mxCell id="1" parent="0"/>
    <mxCell id="n1" value="Input: shape=784" vertex="1" parent="1"/>
    <mxCell id="n2" value="Linear: 256" vertex="1" parent="1"/>
    <mxCell id="e1" edge="1" source="n1" target="n2" parent="1"/>
  </root>
</mxGraphModel>
```

---

## Intermediate Representation (IR)

The IR is a JSON directed graph:

```json
{
  "nodes": [
    {"id": "A", "type": "Input", "params": {"shape": 784}},
    {"id": "B", "type": "Linear", "params": {"out_features": 256}},
    {"id": "C", "type": "ReLU", "params": {}},
    {"id": "D", "type": "Linear", "params": {"out_features": 10}},
    {"id": "E", "type": "Softmax", "params": {}}
  ],
  "edges": [
    {"from": "A", "to": "B"},
    {"from": "B", "to": "C"},
    {"from": "C", "to": "D"},
    {"from": "D", "to": "E"}
  ]
}
```

---

## Optional: Ollama Integration

If you have [Ollama](https://ollama.ai) running locally, the app can use it to polish generated code with comments and formatting. This is entirely optional — the pipeline works fully without it.

```bash
# Install and start Ollama (separate process)
ollama pull codellama
ollama serve
```

Then set `"polish": true` in the `/generate` request.

---

## Architecture Principles

1. **Deterministic logic first** — parsing, validation, and code generation are all rule-based. No LLM controls logic.
2. **LLM is optional** — Ollama integration is best-effort; the system falls back gracefully.
3. **Every module independently testable** — each component has its own test file.
4. **Every stage produces verifiable artifacts** — IR JSON, generated code string, validation results.
5. **Fail gracefully** — specific exception types with descriptive messages at every layer.

---

## Tech Stack

- **Backend**: Python 3.11+, FastAPI, Pydantic
- **ML Frameworks**: PyTorch, TensorFlow/Keras
- **LLM**: Ollama (optional, local)
- **Testing**: pytest, pytest-asyncio, httpx
- **Frontend**: Vanilla HTML/CSS/JS (no build step)
