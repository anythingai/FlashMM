# FlashMM — Predictive On‑Chain Market‑Making Agent for **Sei**

## 1. Vision & Overview
Build an AI‑driven, latency‑sensitive agent that streams Sei order‑book data, predicts short‑term price movement, and automatically places/updates passive quotes to tighten spreads, deepen liquidity, and earn maker fees. The agent should run fully permissionlessly, publish live metrics on‑chain and to social media, and be safe enough to operate unattended during the hackathon demo.

## 2. Goals & Non‑Goals
| | In‑Scope Goals | Out‑of‑Scope (v1) |
|---|---|---|
| **G1** | Provide predictive quoting on at least two spot markets (e.g., SEI/USDC, wETH/USDC) on Sei V2 testnet | Cross‑margining across multiple CLOB venues |
| **G2** | Demonstrate spread improvement ≥ 40 % vs baseline passive book during a 1‑hour window | Derivatives or perpetual contracts |
| **G3** | Maintain net inventory within ±2 % of notional traded | Options market‑making |
| **G4** | Publish real‑time PnL & spread metrics to a public dashboard & X feed | Production‑ready auto‑scaling infrastructure |
| **G5** | Open‑source the core quoting logic under MIT | Commercial SaaS offering |

## 3. Value Proposition
* **For traders:** Tighter spreads → lower slippage.
* **For protocols:** Deeper liquidity → higher volume & revenue.
* **For Sei:** Showcases sub‑second finality & CLOB as ideal for ML‑powered agents.

## 4. Target Users & Stakeholders
* Retail & pro traders using Sei DEXes.
* DeFi protocols seeking liquidity.
* Sei Foundation & Hackathon judges.

## 5. Problem Statement
  > On emerging L1s, wide spreads and shallow books deter volume. Manual market‑making cannot react fast enough; naive bots ignore predictive signals, leading to toxic flow or inventory risk.

## 6. Key Features / Functional Requirements
1. **Real‑Time Data Ingestion**  
   * Subscribe to Sei’s WebSocket order‑book + trade feed (< 250 ms latency target).
2. **Prediction Engine**  
   * Lightweight transformer/LSTM forecasting 100–500 ms mid‑price.
3. **Quote Generation**  
   * Dynamic spread and size based on predicted move & confidence.
4. **Inventory & Risk Management**  
   * Skew quotes to mean‑revert inventory; cut exposure on volatility spikes.
5. **Order Routing & Reconciliation**  
   * Use Cambrian Python SDK to place/modify/cancel orders; ensure deterministic cleanup on disconnect.
6. **Monitoring & Telemetry**  
   * Record fills, PnL, spread, latency; push to Grafana & tweet hourly summary.
7. **Failsafes**  
   * Kill‑switch on model drift or extreme market conditions (> 5 σ price move in 1 s).

## 7. User Stories
* **MM‑01**: *As a DEX trader*, I see narrower spreads (< 0.1 %) so my swaps are cheaper.
* **MM‑02**: *As a hackathon judge*, I view a dashboard proving spread tightening in real time.
* **MM‑03**: *As a risk manager bot*, I automatically widen quotes when inventory > 2 % notional.

## 8. User Flow (High‑Level)
1. Agent starts ➜ streams book & trades ➜ normalizes data.
2. Every 200 ms: model predicts Δprice ➜ quoting module computes bid/ask.
3. Orders sent ➜ fills recorded ➜ inventory updated.
4. Metrics logged ➜ dashboard & X update every 60 s.

## 9. Architecture
```
┌─────────────┐        WebSocket        ┌──────────────┐
│ Sei CLOB    │────────────────────────▶│ Data Ingest  │
└─────────────┘                         └────┬─────────┘
                                            ▼
                                     ┌────────────┐
                                     │ ML Model   │ (PyTorch)
                                     └────┬───────┘
                                          ▼
                                   ┌────────────┐
                                   │ Quoting    │
                                   └────┬───────┘
                                          ▼ REST/WS
                                ┌────────────────────┐
                                │ Order Router (SDK) │
                                └────────────────────┘
                                          │ fills
                                          ▼
                                   ┌────────────┐
                                   │ Telemetry  │→ InfluxDB/Grafana
                                   └────────────┘
```

| Layer | Tech | Notes |
|-------|------|-------|
| ML | PyTorch + TorchScript | Model size < 5 MB, inference < 5 ms |
| Backend | FastAPI | Health & metrics endpoints |
| Infra | Docker‑Compose | Single VPS; latency < 100 ms from Sei RPC |
| Dashboard | Grafana Cloud | Public link |

## 10. Technical Requirements
* **Sei v2 testnet RPC**: `<https://sei-testnet-endpoint>` latency < 150 ms.
* **SDK**: Cambrian Python 0.4+ (supports CLOB v2).
* **Model Training Data**: 48 h historic order‑book snapshots (Sei archive node) + synthetic noise.
* **Inference Frequency**: 5 Hz.
* **Max Quote Depth**: 3 price levels each side.
* **Inventory Limit**: ±2000 USDC per market.
* **Gas Budget**: ≤ 0.05 SEI per 1k orders.

## 11. KPIs & Success Metrics
| Metric | Target |
|--------|--------|
| Average Bid‑Ask Spread | ≥ 40 % narrower vs pre‑deploy baseline |
| Maker Fee Earned | ≥ 0.2 % daily ROI on notional |
| Prediction Accuracy (hit‑rate) | > 55 % direction accuracy @ 100 ms horizon |
| Agent Uptime | > 98 % during judging window |

## 12. Milestones & Timeline (Hackathon ~4 weeks)
| Week | Deliverable |
|------|-------------|
| 1 | Data pipeline + baseline quoting bot |
| 2 | Prototype ML model; back‑test on historic data |
| 3 | Live testnet deployment; Grafana dashboard |
| 4 | Optimization, social feed, demo video |

## 13. Dependencies
* Sei v2 testnet stability.
* Cambrian SDK & RPC quota.
* VPS on AWS (ap‑southeast‑1) with public IP.
* Grafana Cloud free tier.

## 14. Risks & Mitigations
| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Model over‑fits & posts stale quotes | Med | High | Confidence threshold; fallback logic |
| Network congestion delays cancels | Low | Med | Post wider quotes; auto‑widen on latency spike |
| Sei RPC downtime | Low | High | Redundant endpoints; auto‑pause trading |

## 15. Security & Compliance
* Use read‑only keys for data; separate hot key with limited funds for trading.
* Auto‑revoke keys after hackathon.
* Adhere to Sei’s rate limits and Terms.

## 16. Licensing & Open‑Source
* Code released under MIT.
* Model weights & dataset scripts in `/research` folder under CC‑BY‑SA‑4.0.

## 17. Post‑Hack Roadmap
1. Extend to mainnet & additional markets.
2. Incorporate social‑sentiment skew as secondary signal.
3. Multi‑venue routing (Sei + Osmosis order‑book).
4. Governance token incentives for community liquidity vaults.

## 18. Acceptance Criteria
* ✅ Agent running on Sei testnet during live demo.
* ✅ Public dashboard shows at least 30 min of improved spreads.
* ✅ Code repo + README + PRD.md submitted before deadline.

## 19. Appendix
* **Sei CLOB API Docs**: <https://docs.sei.io/core/clob>
* **Cambrian SDK Guide**: <https://github.com/cambrian-code/python-sdk>
* **Example Order Message**:
  ```json
  {
    "market_id": "SEI-USDC",
    "side": "BID",
    "price": "0.0421",
    "size": "500"
  }
  ```

