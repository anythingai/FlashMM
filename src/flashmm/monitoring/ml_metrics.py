"""
FlashMM ML Performance Monitoring

Specialized monitoring and metrics collection for ML prediction engine
with Azure OpenAI cost tracking and performance analytics.
"""

import asyncio
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json

from flashmm.utils.logging import get_logger
from flashmm.monitoring.telemetry.metrics_collector import MetricsCollector
from flashmm.ml.models.prediction_models import PredictionResult, PredictionMethod

logger = get_logger(__name__)


@dataclass
class PredictionMetrics:
    """Metrics for a single prediction."""
    timestamp: datetime
    symbol: str
    method: str
    direction: str
    confidence: float
    response_time_ms: float
    api_success: bool
    validation_passed: bool
    cache_hit: bool
    ensemble_agreement: float
    uncertainty_score: float
    cost_estimate_usd: float = 0.0


@dataclass
class PerformanceWindow:
    """Performance metrics for a time window."""
    start_time: datetime
    end_time: datetime
    total_predictions: int = 0
    successful_predictions: int = 0
    api_successes: int = 0
    cache_hits: int = 0
    
    # Latency metrics
    avg_response_time_ms: float = 0.0
    p95_response_time_ms: float = 0.0
    p99_response_time_ms: float = 0.0
    max_response_time_ms: float = 0.0
    
    # Accuracy metrics (requires ground truth)
    accuracy_samples: int = 0
    directional_accuracy: float = 0.0
    confidence_calibration: float = 0.0
    
    # Method distribution
    method_counts: Dict[str, int] = field(default_factory=dict)
    
    # Cost metrics
    total_cost_usd: float = 0.0
    cost_per_prediction_usd: float = 0.0
    
    # Quality metrics
    avg_confidence: float = 0.0
    avg_ensemble_agreement: float = 0.0
    avg_uncertainty: float = 0.0


class MLMetricsCollector:
    """Specialized metrics collector for ML predictions."""
    
    def __init__(self, window_size_minutes: int = 10, max_windows: int = 144):
        """Initialize ML metrics collector.
        
        Args:
            window_size_minutes: Size of performance windows in minutes
            max_windows: Maximum windows to keep (144 = 24 hours at 10min windows)
        """
        self.window_size_minutes = window_size_minutes 
        self.max_windows = max_windows
        
        # Current metrics storage
        self.current_predictions: List[PredictionMetrics] = []
        self.performance_windows: deque = deque(maxlen=max_windows)
        
        # Real-time metrics
        self.prediction_count = 0
        self.total_cost = 0.0
        self.last_prediction_time: Optional[datetime] = None
        
        # Performance tracking
        self.response_times: deque = deque(maxlen=1000)  # Last 1000 predictions
        self.confidence_scores: deque = deque(maxlen=1000)
        self.method_counters = defaultdict(int)
        
        # Cost tracking
        self.hourly_costs: Dict[str, float] = {}  # Hour -> cost
        self.daily_costs: Dict[str, float] = {}   # Date -> cost
        
        # Alert thresholds
        self.latency_threshold_ms = 500.0
        self.accuracy_threshold = 0.55
        self.cost_alert_threshold_hourly = 10.0  # USD per hour
        
        self._lock = asyncio.Lock()
    
    async def record_prediction(self, prediction_result: PredictionResult, cost_usd: float = 0.0) -> None:
        """Record a prediction for metrics collection.
        
        Args:
            prediction_result: Prediction result to record
            cost_usd: Estimated cost of this prediction in USD
        """
        async with self._lock:
            try:
                # Create metrics record
                metrics = PredictionMetrics(
                    timestamp=prediction_result.timestamp,
                    symbol=prediction_result.symbol,
                    method=prediction_result.method.value,
                    direction=prediction_result.direction,
                    confidence=prediction_result.confidence,
                    response_time_ms=prediction_result.response_time_ms,
                    api_success=prediction_result.api_success,
                    validation_passed=prediction_result.validation_passed,
                    cache_hit=prediction_result.cache_hit,
                    ensemble_agreement=prediction_result.ensemble_agreement,
                    uncertainty_score=prediction_result.uncertainty_score,
                    cost_estimate_usd=cost_usd
                )
                
                # Add to current batch
                self.current_predictions.append(metrics)
                
                # Update real-time metrics
                self.prediction_count += 1
                self.total_cost += cost_usd
                self.last_prediction_time = prediction_result.timestamp
                
                # Update rolling windows
                self.response_times.append(prediction_result.response_time_ms)
                self.confidence_scores.append(prediction_result.confidence)
                self.method_counters[prediction_result.method.value] += 1
                
                # Update cost tracking
                await self._update_cost_tracking(prediction_result.timestamp, cost_usd)
                
                # Check for alerts
                await self._check_alerts(metrics)
                
                # Aggregate window if needed
                await self._maybe_aggregate_window()
                
            except Exception as e:
                logger.error(f"Failed to record prediction metrics: {e}")
    
    async def _update_cost_tracking(self, timestamp: datetime, cost_usd: float) -> None:
        """Update hourly and daily cost tracking."""
        hour_key = timestamp.strftime("%Y-%m-%d-%H")
        date_key = timestamp.strftime("%Y-%m-%d")
        
        self.hourly_costs[hour_key] = self.hourly_costs.get(hour_key, 0.0) + cost_usd
        self.daily_costs[date_key] = self.daily_costs.get(date_key, 0.0) + cost_usd
        
        # Cleanup old entries (keep last 7 days)
        cutoff_date = (timestamp - timedelta(days=7)).strftime("%Y-%m-%d")
        self.daily_costs = {k: v for k, v in self.daily_costs.items() if k >= cutoff_date}
        
        cutoff_hour = (timestamp - timedelta(hours=48)).strftime("%Y-%m-%d-%H")
        self.hourly_costs = {k: v for k, v in self.hourly_costs.items() if k >= cutoff_hour}
    
    async def _check_alerts(self, metrics: PredictionMetrics) -> None:
        """Check for alert conditions."""
        alerts = []
        
        # Latency alert
        if metrics.response_time_ms > self.latency_threshold_ms:
            alerts.append(f"High latency: {metrics.response_time_ms:.1f}ms > {self.latency_threshold_ms}ms")
        
        # Cost alert (check current hour)
        current_hour = metrics.timestamp.strftime("%Y-%m-%d-%H")
        hourly_cost = self.hourly_costs.get(current_hour, 0.0)
        if hourly_cost > self.cost_alert_threshold_hourly:
            alerts.append(f"High hourly cost: ${hourly_cost:.2f} > ${self.cost_alert_threshold_hourly}")
        
        # API failure alert
        if not metrics.api_success and metrics.method == PredictionMethod.AZURE_OPENAI.value:
            alerts.append("Azure OpenAI API failure")
        
        # Log alerts
        for alert in alerts:
            logger.warning(f"ML Metrics Alert [{metrics.symbol}]: {alert}")
    
    async def _maybe_aggregate_window(self) -> None:
        """Aggregate current predictions into performance window if time elapsed."""
        if not self.current_predictions:
            return
        
        # Check if window should be aggregated
        oldest_prediction = self.current_predictions[0].timestamp
        window_age_minutes = (datetime.utcnow() - oldest_prediction).total_seconds() / 60
        
        if window_age_minutes >= self.window_size_minutes:
            await self._aggregate_current_window()
    
    async def _aggregate_current_window(self) -> None:
        """Aggregate current predictions into a performance window."""
        if not self.current_predictions:
            return
        
        try:
            # Calculate window metrics
            start_time = self.current_predictions[0].timestamp
            end_time = self.current_predictions[-1].timestamp
            
            total_predictions = len(self.current_predictions)
            successful_predictions = sum(1 for p in self.current_predictions if p.validation_passed)
            api_successes = sum(1 for p in self.current_predictions if p.api_success)
            cache_hits = sum(1 for p in self.current_predictions if p.cache_hit)
            
            # Latency metrics
            response_times = [p.response_time_ms for p in self.current_predictions]
            response_times.sort()
            
            avg_response_time_ms = sum(response_times) / len(response_times)
            p95_idx = int(len(response_times) * 0.95)
            p99_idx = int(len(response_times) * 0.99)
            p95_response_time_ms = response_times[p95_idx] if p95_idx < len(response_times) else response_times[-1]
            p99_response_time_ms = response_times[p99_idx] if p99_idx < len(response_times) else response_times[-1]
            max_response_time_ms = max(response_times)
            
            # Method distribution
            method_counts = defaultdict(int)
            for p in self.current_predictions:
                method_counts[p.method] += 1
            
            # Cost metrics
            total_cost_usd = sum(p.cost_estimate_usd for p in self.current_predictions)
            cost_per_prediction_usd = total_cost_usd / total_predictions if total_predictions > 0 else 0.0
            
            # Quality metrics
            avg_confidence = sum(p.confidence for p in self.current_predictions) / total_predictions
            avg_ensemble_agreement = sum(p.ensemble_agreement for p in self.current_predictions) / total_predictions
            avg_uncertainty = sum(p.uncertainty_score for p in self.current_predictions) / total_predictions
            
            # Create performance window
            window = PerformanceWindow(
                start_time=start_time,
                end_time=end_time,
                total_predictions=total_predictions,
                successful_predictions=successful_predictions,
                api_successes=api_successes,
                cache_hits=cache_hits,
                avg_response_time_ms=avg_response_time_ms,
                p95_response_time_ms=p95_response_time_ms,
                p99_response_time_ms=p99_response_time_ms,
                max_response_time_ms=max_response_time_ms,
                method_counts=dict(method_counts),
                total_cost_usd=total_cost_usd,
                cost_per_prediction_usd=cost_per_prediction_usd,
                avg_confidence=avg_confidence,
                avg_ensemble_agreement=avg_ensemble_agreement,
                avg_uncertainty=avg_uncertainty
            )
            
            # Add to windows
            self.performance_windows.append(window)
            
            # Clear current predictions
            self.current_predictions.clear()
            
            logger.info(f"Aggregated ML performance window: {total_predictions} predictions, "
                       f"avg latency {avg_response_time_ms:.1f}ms, cost ${total_cost_usd:.4f}")
            
        except Exception as e:
            logger.error(f"Failed to aggregate performance window: {e}")
    
    async def get_current_performance(self) -> Dict[str, Any]:
        """Get current real-time performance metrics."""
        async with self._lock:
            try:
                # Current response time percentiles
                response_times_list = list(self.response_times)
                if response_times_list:
                    response_times_list.sort()
                    p50_latency = response_times_list[len(response_times_list) // 2]
                    p95_latency = response_times_list[int(len(response_times_list) * 0.95)]
                    p99_latency = response_times_list[int(len(response_times_list) * 0.99)]
                    avg_latency = sum(response_times_list) / len(response_times_list)
                else:
                    p50_latency = p95_latency = p99_latency = avg_latency = 0.0
                
                # Current confidence stats
                confidence_list = list(self.confidence_scores)
                avg_confidence = sum(confidence_list) / len(confidence_list) if confidence_list else 0.0
                
                # Current hour cost
                current_hour = datetime.utcnow().strftime("%Y-%m-%d-%H")
                current_hour_cost = self.hourly_costs.get(current_hour, 0.0)
                
                # Today's cost
                today = datetime.utcnow().strftime("%Y-%m-%d")
                today_cost = self.daily_costs.get(today, 0.0)
                
                return {
                    'timestamp': datetime.utcnow().isoformat(),
                    'prediction_count': self.prediction_count,
                    'total_cost_usd': self.total_cost,
                    'last_prediction_time': self.last_prediction_time.isoformat() if self.last_prediction_time else None,
                    
                    'latency': {
                        'avg_ms': avg_latency,
                        'p50_ms': p50_latency,
                        'p95_ms': p95_latency,
                        'p99_ms': p99_latency,
                        'samples': len(response_times_list)
                    },
                    
                    'quality': {
                        'avg_confidence': avg_confidence,
                        'confidence_samples': len(confidence_list)
                    },
                    
                    'methods': dict(self.method_counters),
                    
                    'costs': {
                        'current_hour_usd': current_hour_cost,
                        'today_usd': today_cost,
                        'total_usd': self.total_cost
                    },
                    
                    'pending_predictions': len(self.current_predictions),
                    'performance_windows': len(self.performance_windows)
                }
                
            except Exception as e:
                logger.error(f"Failed to get current performance: {e}")
                return {'error': str(e), 'timestamp': datetime.utcnow().isoformat()}
    
    async def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance summary for the last N hours."""
        async with self._lock:
            try:
                cutoff_time = datetime.utcnow() - timedelta(hours=hours)
                
                # Filter recent windows
                recent_windows = [w for w in self.performance_windows if w.end_time >= cutoff_time]
                
                if not recent_windows:
                    return {
                        'period_hours': hours,
                        'windows': 0,
                        'total_predictions': 0,
                        'summary': 'No data available'
                    }
                
                # Aggregate metrics
                total_predictions = sum(w.total_predictions for w in recent_windows)
                successful_predictions = sum(w.successful_predictions for w in recent_windows)
                api_successes = sum(w.api_successes for w in recent_windows)
                cache_hits = sum(w.cache_hits for w in recent_windows)
                total_cost = sum(w.total_cost_usd for w in recent_windows)
                
                # Weighted averages
                if total_predictions > 0:
                    weighted_avg_latency = sum(w.avg_response_time_ms * w.total_predictions for w in recent_windows) / total_predictions
                    weighted_avg_confidence = sum(w.avg_confidence * w.total_predictions for w in recent_windows) / total_predictions 
                    weighted_avg_agreement = sum(w.avg_ensemble_agreement * w.total_predictions for w in recent_windows) / total_predictions
                else:
                    weighted_avg_latency = weighted_avg_confidence = weighted_avg_agreement = 0.0
                
                # Method distribution
                method_totals = defaultdict(int)
                for window in recent_windows:
                    for method, count in window.method_counts.items():
                        method_totals[method] += count
                
                # Performance percentiles across windows
                all_p95_latencies = [w.p95_response_time_ms for w in recent_windows]
                all_p99_latencies = [w.p99_response_time_ms for w in recent_windows]
                max_p95_latency = max(all_p95_latencies) if all_p95_latencies else 0.0
                max_p99_latency = max(all_p99_latencies) if all_p99_latencies else 0.0
                
                return {
                    'period_hours': hours,
                    'windows': len(recent_windows),
                    'start_time': recent_windows[0].start_time.isoformat(),
                    'end_time': recent_windows[-1].end_time.isoformat(),
                    
                    'totals': {
                        'predictions': total_predictions,
                        'successful': successful_predictions,
                        'api_successes': api_successes,
                        'cache_hits': cache_hits,
                        'cost_usd': total_cost
                    },
                    
                    'rates': {
                        'success_rate': successful_predictions / max(total_predictions, 1),
                        'api_success_rate': api_successes / max(total_predictions, 1),
                        'cache_hit_rate': cache_hits / max(total_predictions, 1),
                        'predictions_per_hour': total_predictions / max(hours, 1)
                    },
                    
                    'performance': {
                        'avg_latency_ms': weighted_avg_latency,
                        'max_p95_latency_ms': max_p95_latency,
                        'max_p99_latency_ms': max_p99_latency,
                        'avg_confidence': weighted_avg_confidence,
                        'avg_ensemble_agreement': weighted_avg_agreement
                    },
                    
                    'methods': dict(method_totals),
                    
                    'costs': {
                        'total_usd': total_cost,
                        'avg_per_prediction_usd': total_cost / max(total_predictions, 1),
                        'hourly_rate_usd': total_cost / max(hours, 1)
                    }
                }
                
            except Exception as e:
                logger.error(f"Failed to get performance summary: {e}")
                return {'error': str(e), 'period_hours': hours}
    
    async def record_accuracy_sample(self, 
                                   prediction_id: str,
                                   predicted_direction: str,
                                   actual_direction: str,
                                   predicted_confidence: float) -> None:
        """Record accuracy sample for model evaluation.
        
        Args:
            prediction_id: Unique prediction identifier
            predicted_direction: Predicted price direction
            actual_direction: Actual price direction
            predicted_confidence: Prediction confidence
        """
        # This would be used to track prediction accuracy over time
        # Implementation would store samples and calculate accuracy metrics
        try:
            is_correct = predicted_direction == actual_direction
            
            # Log accuracy sample (in production, store in database)
            logger.info(f"Accuracy sample: {prediction_id} - "
                       f"Predicted: {predicted_direction} ({predicted_confidence:.3f}), "
                       f"Actual: {actual_direction}, Correct: {is_correct}")
            
            # Update accuracy tracking (simplified)
            # In production, this would update rolling accuracy windows
            
        except Exception as e:
            logger.error(f"Failed to record accuracy sample: {e}")
    
    async def get_cost_breakdown(self) -> Dict[str, Any]:
        """Get detailed cost breakdown."""
        async with self._lock:
            try:
                # Last 24 hours by hour
                last_24h = {}
                now = datetime.utcnow()
                for i in range(24):
                    hour_time = now - timedelta(hours=i)
                    hour_key = hour_time.strftime("%Y-%m-%d-%H")
                    last_24h[hour_key] = self.hourly_costs.get(hour_key, 0.0)
                
                # Last 7 days by day
                last_7d = {}
                for i in range(7):
                    day_time = now - timedelta(days=i)
                    day_key = day_time.strftime("%Y-%m-%d")
                    last_7d[day_key] = self.daily_costs.get(day_key, 0.0)
                
                # Projections
                current_hour_cost = self.hourly_costs.get(now.strftime("%Y-%m-%d-%H"), 0.0)
                projected_daily = current_hour_cost * 24
                projected_monthly = sum(last_7d.values()) / 7 * 30
                
                return {
                    'timestamp': now.isoformat(),
                    'hourly_breakdown': last_24h,
                    'daily_breakdown': last_7d,
                    'totals': {
                        'last_24h_usd': sum(last_24h.values()),
                        'last_7d_usd': sum(last_7d.values()),
                        'total_tracked_usd': self.total_cost
                    },
                    'projections': {
                        'daily_usd': projected_daily,
                        'monthly_usd': projected_monthly
                    }
                }
                
            except Exception as e:
                logger.error(f"Failed to get cost breakdown: {e}")
                return {'error': str(e)}
    
    async def cleanup_old_data(self) -> None:
        """Clean up old performance data."""
        async with self._lock:
            try:
                # Cleanup is automatic via deque maxlen for windows
                
                # Clean old cost data (keep 30 days)
                cutoff_date = (datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%d")
                old_daily_keys = [k for k in self.daily_costs.keys() if k < cutoff_date]
                for key in old_daily_keys:
                    del self.daily_costs[key]
                
                cutoff_hour = (datetime.utcnow() - timedelta(hours=72)).strftime("%Y-%m-%d-%H")
                old_hourly_keys = [k for k in self.hourly_costs.keys() if k < cutoff_hour]
                for key in old_hourly_keys:
                    del self.hourly_costs[key]
                
                logger.info(f"Cleaned up {len(old_daily_keys)} daily and {len(old_hourly_keys)} hourly cost entries")
                
            except Exception as e:
                logger.error(f"Failed to cleanup old data: {e}")


class MLPerformanceDashboard:
    """Performance dashboard for ML metrics visualization."""
    
    def __init__(self, metrics_collector: MLMetricsCollector):
        """Initialize dashboard.
        
        Args:
            metrics_collector: ML metrics collector instance
        """
        self.metrics_collector = metrics_collector
    
    async def generate_performance_report(self, hours: int = 24) -> str:
        """Generate human-readable performance report.
        
        Args:
            hours: Hours to include in report
            
        Returns:
            Formatted performance report
        """
        try:
            current_perf = await self.metrics_collector.get_current_performance()
            summary = await self.metrics_collector.get_performance_summary(hours)
            cost_breakdown = await self.metrics_collector.get_cost_breakdown()
            
            report_lines = [
                f"=== FlashMM ML Performance Report ===",
                f"Generated: {datetime.utcnow().isoformat()}",
                f"Period: Last {hours} hours",
                "",
                "📊 CURRENT PERFORMANCE:",
                f"  • Total Predictions: {current_perf['prediction_count']:,}",
                f"  • Avg Latency: {current_perf['latency']['avg_ms']:.1f}ms",
                f"  • P95 Latency: {current_perf['latency']['p95_ms']:.1f}ms",
                f"  • P99 Latency: {current_perf['latency']['p99_ms']:.1f}ms",
                f"  • Avg Confidence: {current_perf['quality']['avg_confidence']:.3f}",
                "",
                "📈 PERIOD SUMMARY:",
                f"  • Predictions: {summary['totals']['predictions']:,}",
                f"  • Success Rate: {summary['rates']['success_rate']:.1%}",
                f"  • API Success Rate: {summary['rates']['api_success_rate']:.1%}",
                f"  • Cache Hit Rate: {summary['rates']['cache_hit_rate']:.1%}",
                f"  • Predictions/Hour: {summary['rates']['predictions_per_hour']:.1f}",
                "",
                "⚡ PERFORMANCE:",
                f"  • Avg Latency: {summary['performance']['avg_latency_ms']:.1f}ms",
                f"  • Max P95 Latency: {summary['performance']['max_p95_latency_ms']:.1f}ms",
                f"  • Avg Confidence: {summary['performance']['avg_confidence']:.3f}",
                f"  • Avg Agreement: {summary['performance']['avg_ensemble_agreement']:.3f}",
                "",
                "💰 COSTS:",
                f"  • Total Cost: ${summary['costs']['total_usd']:.4f}",
                f"  • Per Prediction: ${summary['costs']['avg_per_prediction_usd']:.6f}",
                f"  • Hourly Rate: ${summary['costs']['hourly_rate_usd']:.4f}",
                f"  • Current Hour: ${current_perf['costs']['current_hour_usd']:.4f}",
                f"  • Today: ${current_perf['costs']['today_usd']:.4f}",
                "",
                "🔧 METHODS:",
            ]
            
            for method, count in summary['methods'].items():
                percentage = count / max(summary['totals']['predictions'], 1) * 100
                report_lines.append(f"  • {method}: {count:,} ({percentage:.1f}%)")
            
            return "\n".join(report_lines)
            
        except Exception as e:
            logger.error(f"Failed to generate performance report: {e}")
            return f"Error generating report: {e}"


# Global instance
_ml_metrics_collector: Optional[MLMetricsCollector] = None


def get_ml_metrics_collector() -> MLMetricsCollector:
    """Get global ML metrics collector instance."""
    global _ml_metrics_collector
    if _ml_metrics_collector is None:
        _ml_metrics_collector = MLMetricsCollector()
    return _ml_metrics_collector