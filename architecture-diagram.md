# FlashMM Technical Architecture & Data Flow

## System Architecture Overview

```mermaid
graph TB
    %% External Systems
    SEI[Sei Blockchain CLOB v2]
    GRAFANA[Grafana Cloud Dashboard]
    TWITTER[X/Twitter API]
    CLIENT[Dashboard Clients]
    
    %% Data Ingestion Layer
    subgraph "Data Ingestion Layer"
        WS[WebSocket Client]
        NORM[Data Normalizer]
        FEED[Feed Manager]
    end
    
    %% Storage Layer
    subgraph "Storage & Cache Layer"
        REDIS[(Redis Cache)]
        INFLUX[(InfluxDB)]
    end
    
    %% ML Pipeline
    subgraph "ML Inference Pipeline"
        PREP[Feature Engineering]
        MODEL[TorchScript Model]
        PRED[Prediction Engine]
    end
    
    %% Trading Engine
    subgraph "Trading Engine"
        QUOTE[Quote Generator]
        RISK[Risk Manager]
        EXEC[Order Executor]
        ROUTER[Cambrian SDK Router]
    end
    
    %% API & Monitoring
    subgraph "API & Monitoring Layer"
        API[FastAPI Server]
        METRICS[Metrics Collector]
        TELEM[Telemetry Service]
        SOCIAL[Social Publisher]
    end
    
    %% Data Flow Connections
    SEI -.->|WebSocket Stream| WS
    WS --> NORM
    NORM --> FEED
    FEED --> REDIS
    FEED --> PREP
    
    REDIS --> PREP
    PREP --> MODEL
    MODEL --> PRED
    PRED --> QUOTE
    
    QUOTE --> RISK
    RISK --> EXEC
    EXEC --> ROUTER
    ROUTER -.->|Orders/Cancels| SEI
    SEI -.->|Fill Confirmations| ROUTER
    
    ROUTER --> METRICS
    PRED --> METRICS
    QUOTE --> METRICS
    RISK --> METRICS
    
    METRICS --> TELEM
    TELEM --> INFLUX
    TELEM --> API
    TELEM --> SOCIAL
    
    SOCIAL -.->|Performance Updates| TWITTER
    INFLUX -.->|Dashboard Data| GRAFANA
    API -.->|REST/WebSocket| CLIENT
    
    %% Styling
    classDef external fill:#e1f5fe
    classDef storage fill:#f3e5f5
    classDef ml fill:#e8f5e8
    classDef trading fill:#fff3e0
    classDef api fill:#fce4ec
    
    class SEI,GRAFANA,TWITTER,CLIENT external
    class REDIS,INFLUX storage
    class PREP,MODEL,PRED ml
    class QUOTE,RISK,EXEC,ROUTER trading
    class API,METRICS,TELEM,SOCIAL api
```

## Detailed Component Data Flow

### 1. Real-Time Data Pipeline

```mermaid
sequenceDiagram
    participant SEI as Sei CLOB
    participant WS as WebSocket Client
    participant NORM as Data Normalizer
    participant REDIS as Redis Cache
    participant PREP as Feature Engineering
    
    SEI->>WS: Order book updates (< 250ms)
    SEI->>WS: Trade executions
    WS->>NORM: Raw market data
    NORM->>REDIS: Normalized order book
    NORM->>PREP: Structured features
    REDIS->>PREP: Historical snapshots
    Note over PREP: 5Hz processing cycle
```

### 2. ML Inference Pipeline

```mermaid
sequenceDiagram
    participant PREP as Feature Engineering
    participant MODEL as TorchScript Model
    participant PRED as Prediction Engine
    participant QUOTE as Quote Generator
    
    PREP->>MODEL: Feature vectors (200ms window)
    MODEL->>PRED: Price movement predictions
    Note over MODEL: Inference < 5ms
    PRED->>QUOTE: Direction + Confidence
    Note over QUOTE: Generate bid/ask spreads
```

### 3. Trading Execution Flow

```mermaid
sequenceDiagram
    participant QUOTE as Quote Generator
    participant RISK as Risk Manager
    participant EXEC as Order Executor
    participant SDK as Cambrian SDK
    participant SEI as Sei CLOB
    
    QUOTE->>RISK: Proposed quotes
    RISK->>RISK: Check inventory limits
    RISK->>RISK: Validate position size
    RISK->>EXEC: Approved orders
    EXEC->>SDK: Place/modify orders
    SDK->>SEI: Submit transactions
    SEI->>SDK: Fill confirmations
    SDK->>EXEC: Execution updates
    EXEC->>RISK: Update positions
```

### 4. Monitoring & Telemetry Flow

```mermaid
sequenceDiagram
    participant COMPONENTS as All Components
    participant METRICS as Metrics Collector
    participant TELEM as Telemetry Service
    participant INFLUX as InfluxDB
    participant API as FastAPI
    participant SOCIAL as Social Publisher
    
    COMPONENTS->>METRICS: Performance metrics
    METRICS->>TELEM: Aggregated data
    TELEM->>INFLUX: Time-series storage
    TELEM->>API: Real-time endpoints
    TELEM->>SOCIAL: Performance summaries
    Note over SOCIAL: Hourly X/Twitter updates
```

## Component Interaction Matrix

| Component | Inputs From | Outputs To | Frequency | Latency Target |
|-----------|-------------|------------|-----------|----------------|
| WebSocket Client | Sei CLOB | Data Normalizer | Real-time | < 250ms |
| Data Normalizer | WebSocket Client | Redis, Feature Engineering | Real-time | < 10ms |
| Feature Engineering | Normalizer, Redis | ML Model | 5Hz | < 20ms |
| ML Model | Feature Engineering | Prediction Engine | 5Hz | < 5ms |
| Quote Generator | Prediction Engine | Risk Manager | 5Hz | < 5ms |
| Risk Manager | Quote Generator, Position Tracker | Order Executor | 5Hz | < 10ms |
| Order Executor | Risk Manager | Cambrian SDK | As needed | < 50ms |
| Metrics Collector | All Components | Telemetry Service | 1Hz | < 100ms |
| Telemetry Service | Metrics Collector | InfluxDB, API, Social | 1Hz | < 200ms |

## Critical Path Analysis

### Hot Path (Latency Critical)
```
Sei WebSocket → Normalizer → Feature Engineering → ML Model → Quote Generator → Risk Manager → Order Executor → Cambrian SDK → Sei CLOB
```
**Total Target Latency: < 350ms end-to-end**

### Cold Path (Analytics)
```
Metrics Collector → Telemetry Service → InfluxDB → Grafana Dashboard
```
**Target Update Frequency: 1Hz**

## Failure Modes & Circuit Breakers

```mermaid
graph TB
    subgraph "Circuit Breakers"
        CB1[WebSocket Disconnect]
        CB2[ML Model Drift]
        CB3[High Latency Detection]
        CB4[Inventory Limit Breach]
        CB5[Extreme Market Conditions]
    end
    
    CB1 --> PAUSE[Pause Trading]
    CB2 --> FALLBACK[Use Fallback Pricing]
    CB3 --> WIDEN[Widen Spreads]
    CB4 --> FLATTEN[Flatten Position]
    CB5 --> KILL[Emergency Stop]
    
    PAUSE --> MONITOR[Monitor & Alert]
    FALLBACK --> MONITOR
    WIDEN --> MONITOR
    FLATTEN --> MONITOR
    KILL --> MONITOR
```

## Performance Requirements Summary

| Metric | Target | Measurement |
|--------|--------|-------------|
| WebSocket Latency | < 250ms | Round-trip ping |
| ML Inference Time | < 5ms | Model forward pass |
| Order Placement | < 50ms | SDK to blockchain |
| End-to-End Latency | < 350ms | Signal to order |
| System Uptime | > 98% | During demo window |
| Memory Usage | < 2GB | Peak consumption |
| Model Size | < 5MB | TorchScript export |
| Quote Frequency | 5Hz | Orders per second |