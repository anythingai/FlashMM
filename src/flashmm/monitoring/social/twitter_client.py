"""
FlashMM Twitter/X Social Media Integration

Automated social media posting for performance updates, achievements, 
and real-time trading insights with engagement tracking and analytics.
"""

import asyncio
import aiohttp
import json
import base64
import hmac
import hashlib
import urllib.parse
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import os

from flashmm.config.settings import get_config
from flashmm.utils.logging import get_logger

logger = get_logger(__name__)


class PostType(Enum):
    """Types of social media posts."""
    PERFORMANCE_UPDATE = "performance_update"
    ACHIEVEMENT = "achievement"
    ALERT = "alert"
    DAILY_SUMMARY = "daily_summary"
    HOURLY_UPDATE = "hourly_update"
    MILESTONE = "milestone"
    TREND_ANALYSIS = "trend_analysis"


@dataclass
class TwitterConfig:
    """Twitter API configuration."""
    api_key: str
    api_secret: str
    access_token: str
    access_token_secret: str
    bearer_token: str
    client_id: str = ""
    client_secret: str = ""


@dataclass
class PostMetrics:
    """Social media post metrics."""
    post_id: str
    post_type: PostType
    content: str
    timestamp: datetime
    impressions: int = 0
    likes: int = 0
    retweets: int = 0
    replies: int = 0
    engagement_rate: float = 0.0
    reach: int = 0


@dataclass
class TrendData:
    """Trend analysis data."""
    metric_name: str
    current_value: float
    previous_value: float
    change_percent: float
    trend_direction: str  # "up", "down", "stable"
    significance: str  # "high", "medium", "low"


class TwitterClient:
    """Twitter/X API client for automated posting and engagement tracking."""
    
    def __init__(self):
        self.config = get_config()
        
        # Twitter API configuration
        self.twitter_config = self._load_twitter_config()
        
        # API endpoints
        self.base_url = "https://api.twitter.com/2"
        self.upload_url = "https://upload.twitter.com/1.1"
        
        # Session for HTTP requests
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Post templates and content
        self.post_templates = self._initialize_post_templates()
        self.hashtags = self._initialize_hashtags()
        
        # Analytics and tracking
        self.post_history: List[PostMetrics] = []
        self.engagement_stats = {
            "total_posts": 0,
            "total_impressions": 0,
            "total_engagement": 0,
            "avg_engagement_rate": 0.0,
            "best_performing_post": None,
            "trending_topics": []
        }
        
        # Posting schedule and limits
        self.posting_enabled = self.config.get("social.twitter.enabled", False)
        self.rate_limits = {
            "posts_per_hour": 25,
            "posts_per_day": 300,
            "hourly_posts": 0,
            "daily_posts": 0,
            "last_reset_hour": datetime.now().hour,
            "last_reset_date": datetime.now().date()
        }
        
        logger.info(f"TwitterClient initialized (enabled: {self.posting_enabled})")
    
    def _load_twitter_config(self) -> TwitterConfig:
        """Load Twitter API configuration."""
        return TwitterConfig(
            api_key=self.config.get("social.twitter.api_key", ""),
            api_secret=self.config.get("social.twitter.api_secret", ""),
            access_token=self.config.get("social.twitter.access_token", ""),
            access_token_secret=self.config.get("social.twitter.access_token_secret", ""),
            bearer_token=self.config.get("social.twitter.bearer_token", ""),
            client_id=self.config.get("social.twitter.client_id", ""),
            client_secret=self.config.get("social.twitter.client_secret", "")
        )
    
    def _initialize_post_templates(self) -> Dict[PostType, List[str]]:
        """Initialize post templates for different types."""
        return {
            PostType.PERFORMANCE_UPDATE: [
                "🚀 FlashMM Performance Update:\n• Spread Improvement: {spread_improvement:.1f}%\n• P&L: ${pnl:.2f}\n• Volume: ${volume:,.0f}\n• Fill Rate: {fill_rate:.1f}%\n\n{hashtags}",
                "📊 Live Trading Stats:\nSpread: ⬇️ {spread_improvement:.1f}%\nProfit: 💰 ${pnl:.2f}\nVolume: 📈 ${volume:,.0f}\nEfficiency: ⚡ {fill_rate:.1f}%\n\n{hashtags}",
                "⚡ Real-time FlashMM metrics:\n\n🎯 Spread optimization: {spread_improvement:.1f}%\n💹 Current P&L: ${pnl:.2f}\n📊 Trading volume: ${volume:,.0f}\n🔥 Fill rate: {fill_rate:.1f}%\n\n{hashtags}"
            ],
            
            PostType.ACHIEVEMENT: [
                "🎉 Milestone Achievement!\n\nFlashMM just hit {achievement}!\n\n📈 Key metrics:\n• {metric1}: {value1}\n• {metric2}: {value2}\n• {metric3}: {value3}\n\n{hashtags}",
                "🏆 New Record! FlashMM achieved {achievement}\n\nThis marks a significant improvement in our market making performance. Thanks to our advanced ML algorithms! 🤖\n\n{hashtags}",
                "🌟 Breaking News: FlashMM reaches {achievement}!\n\nOur innovative approach to DeFi market making continues to deliver exceptional results.\n\n{hashtags}"
            ],
            
            PostType.DAILY_SUMMARY: [
                "📅 FlashMM Daily Summary ({date}):\n\n📊 Total Volume: ${volume:,.0f}\n💰 P&L: ${pnl:.2f}\n📈 Spread Improvement: {spread:.1f}%\n🎯 Trades: {trades:,}\n⚡ Avg Latency: {latency:.1f}ms\n\n{hashtags}",
                "🌅 Good morning! Here's yesterday's FlashMM performance:\n\n• Volume traded: ${volume:,.0f}\n• Profit generated: ${pnl:.2f}\n• Spread tightened by: {spread:.1f}%\n• Total trades: {trades:,}\n\n{hashtags}",
                "🌙 End of day recap:\n\nFlashMM delivered another strong performance with ${volume:,.0f} in volume and {spread:.1f}% spread improvement.\n\nSee full stats: {dashboard_url}\n\n{hashtags}"
            ],
            
            PostType.HOURLY_UPDATE: [
                "⏰ Hourly Update: FlashMM is delivering {spread:.1f}% spread improvement with ${volume:,.0f} volume in the last hour. {trend_emoji}\n\n{hashtags}",
                "📈 Last hour: {volume_change:+.1f}% volume, {spread_change:+.1f}% spread improvement vs previous hour. {performance_comment}\n\n{hashtags}"
            ],
            
            PostType.ALERT: [
                "🚨 FlashMM Alert: {alert_type}\n\n{alert_message}\n\nCurrent status: {status}\n\n{hashtags}",
                "⚠️ System Alert: {alert_message}\n\nOur team is {action_taken}. All systems monitoring normal.\n\n{hashtags}"
            ],
            
            PostType.MILESTONE: [
                "🎖️ Major Milestone: FlashMM has processed ${total_volume:,.0f} in total volume!\n\nWe've improved spreads by an average of {avg_spread:.1f}% across all trades.\n\n{hashtags}",
                "🌟 Celebration time! FlashMM just completed trade #{trade_count:,} with continued excellence in market making.\n\n{hashtags}"
            ],
            
            PostType.TREND_ANALYSIS: [
                "📊 Trend Analysis: {metric} is {trend_direction} by {change:.1f}% over the last {timeframe}.\n\n{analysis}\n\n{hashtags}",
                "🔍 Market Insight: We're seeing {trend_description} in {metric}. This {impact_description}.\n\nFull analysis: {dashboard_url}\n\n{hashtags}"
            ]
        }
    
    def _initialize_hashtags(self) -> Dict[str, List[str]]:
        """Initialize hashtag collections."""
        return {
            "general": ["#FlashMM", "#DeFi", "#MarketMaking", "#TradingBot", "#Blockchain"],
            "performance": ["#Trading", "#Performance", "#Optimization", "#Spreads", "#Liquidity"],
            "technical": ["#MachineLearning", "#AI", "#Algorithm", "#FinTech", "#Automation"],
            "sei": ["#Sei", "#SeiNetwork", "#SEIUSDC", "#SeiDeFi"],
            "achievements": ["#Milestone", "#Achievement", "#Success", "#Innovation"],
            "alerts": ["#Alert", "#Monitoring", "#SystemHealth", "#RealTime"]
        }
    
    async def initialize(self) -> None:
        """Initialize Twitter client and verify credentials."""
        try:
            if not self.posting_enabled:
                logger.info("Twitter posting disabled in configuration")
                return
            
            if not all([
                self.twitter_config.api_key,
                self.twitter_config.api_secret,
                self.twitter_config.access_token,
                self.twitter_config.access_token_secret
            ]):
                logger.warning("Incomplete Twitter API credentials - posting will be disabled")
                self.posting_enabled = False
                return
            
            # Create HTTP session
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )
            
            # Verify credentials
            await self._verify_credentials()
            
            logger.info("TwitterClient initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize TwitterClient: {e}")
            self.posting_enabled = False
    
    async def _verify_credentials(self) -> bool:
        """Verify Twitter API credentials."""
        try:
            headers = self._create_oauth_headers("GET", f"{self.base_url}/users/me")
            
            async with self.session.get(
                f"{self.base_url}/users/me",
                headers=headers
            ) as response:
                if response.status == 200:
                    user_data = await response.json()
                    username = user_data.get("data", {}).get("username", "unknown")
                    logger.info(f"Twitter credentials verified for @{username}")
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"Twitter credential verification failed: {error_text}")
                    return False
        
        except Exception as e:
            logger.error(f"Error verifying Twitter credentials: {e}")
            return False
    
    def _create_oauth_headers(self, method: str, url: str, params: Dict[str, str] = None) -> Dict[str, str]:
        """Create OAuth 1.0a headers for Twitter API."""
        import time
        import secrets
        
        oauth_params = {
            "oauth_consumer_key": self.twitter_config.api_key,
            "oauth_token": self.twitter_config.access_token,
            "oauth_signature_method": "HMAC-SHA1",
            "oauth_timestamp": str(int(time.time())),
            "oauth_nonce": secrets.token_hex(16),
            "oauth_version": "1.0"
        }
        
        # Add request parameters to OAuth params for signature
        all_params = oauth_params.copy()
        if params:
            all_params.update(params)
        
        # Create signature base string
        sorted_params = sorted(all_params.items())
        param_string = "&".join([f"{k}={urllib.parse.quote(str(v), safe='')}" for k, v in sorted_params])
        base_string = f"{method}&{urllib.parse.quote(url, safe='')}&{urllib.parse.quote(param_string, safe='')}"
        
        # Create signing key
        signing_key = f"{urllib.parse.quote(self.twitter_config.api_secret, safe='')}&{urllib.parse.quote(self.twitter_config.access_token_secret, safe='')}"
        
        # Create signature
        signature = base64.b64encode(
            hmac.new(signing_key.encode(), base_string.encode(), hashlib.sha1).digest()
        ).decode()
        
        oauth_params["oauth_signature"] = signature
        
        # Create authorization header
        auth_header = "OAuth " + ", ".join([f'{k}="{urllib.parse.quote(str(v), safe="")}"' for k, v in sorted(oauth_params.items())])
        
        return {
            "Authorization": auth_header,
            "Content-Type": "application/json"
        }
    
    async def post_performance_update(self, metrics: Dict[str, Any]) -> Optional[PostMetrics]:
        """Post performance update with current metrics."""
        try:
            if not self._can_post():
                return None
            
            # Select template and format content
            template = self._select_template(PostType.PERFORMANCE_UPDATE)
            hashtags = self._format_hashtags(["general", "performance"])
            
            content = template.format(
                spread_improvement=metrics.get("spread_improvement_percent", 0),
                pnl=metrics.get("total_pnl_usdc", 0),
                volume=metrics.get("total_volume_usdc", 0),
                fill_rate=metrics.get("fill_rate_percent", 0),
                hashtags=hashtags
            )
            
            return await self._create_post(content, PostType.PERFORMANCE_UPDATE)
            
        except Exception as e:
            logger.error(f"Failed to post performance update: {e}")
            return None
    
    async def post_achievement(self, achievement: str, metrics: Dict[str, Any]) -> Optional[PostMetrics]:
        """Post achievement announcement."""
        try:
            if not self._can_post():
                return None
            
            template = self._select_template(PostType.ACHIEVEMENT)
            hashtags = self._format_hashtags(["general", "achievements"])
            
            # Get top 3 metrics for achievement post
            metric_items = list(metrics.items())[:3]
            
            content = template.format(
                achievement=achievement,
                metric1=metric_items[0][0] if len(metric_items) > 0 else "N/A",
                value1=metric_items[0][1] if len(metric_items) > 0 else "N/A",
                metric2=metric_items[1][0] if len(metric_items) > 1 else "N/A",
                value2=metric_items[1][1] if len(metric_items) > 1 else "N/A",
                metric3=metric_items[2][0] if len(metric_items) > 2 else "N/A",
                value3=metric_items[2][1] if len(metric_items) > 2 else "N/A",
                hashtags=hashtags
            )
            
            return await self._create_post(content, PostType.ACHIEVEMENT)
            
        except Exception as e:
            logger.error(f"Failed to post achievement: {e}")
            return None
    
    async def post_daily_summary(self, daily_metrics: Dict[str, Any]) -> Optional[PostMetrics]:
        """Post daily summary report."""
        try:
            if not self._can_post():
                return None
            
            template = self._select_template(PostType.DAILY_SUMMARY)
            hashtags = self._format_hashtags(["general", "performance"])
            
            content = template.format(
                date=datetime.now().strftime("%Y-%m-%d"),
                volume=daily_metrics.get("total_volume_usdc", 0),
                pnl=daily_metrics.get("total_pnl_usdc", 0),
                spread=daily_metrics.get("avg_spread_improvement_percent", 0),
                trades=daily_metrics.get("total_trades", 0),
                latency=daily_metrics.get("avg_latency_ms", 0),
                dashboard_url=self.config.get("dashboard.public_url", ""),
                hashtags=hashtags
            )
            
            return await self._create_post(content, PostType.DAILY_SUMMARY)
            
        except Exception as e:
            logger.error(f"Failed to post daily summary: {e}")
            return None
    
    async def post_alert(self, alert_type: str, message: str, status: str = "monitoring") -> Optional[PostMetrics]:
        """Post system alert."""
        try:
            if not self._can_post():
                return None
            
            template = self._select_template(PostType.ALERT)
            hashtags = self._format_hashtags(["general", "alerts"])
            
            content = template.format(
                alert_type=alert_type,
                alert_message=message,
                status=status,
                action_taken="investigating" if "error" in alert_type.lower() else "monitoring",
                hashtags=hashtags
            )
            
            return await self._create_post(content, PostType.ALERT)
            
        except Exception as e:
            logger.error(f"Failed to post alert: {e}")
            return None
    
    async def post_trend_analysis(self, trend_data: TrendData) -> Optional[PostMetrics]:
        """Post trend analysis."""
        try:
            if not self._can_post():
                return None
            
            template = self._select_template(PostType.TREND_ANALYSIS)
            hashtags = self._format_hashtags(["general", "technical"])
            
            # Generate analysis text
            analysis = self._generate_trend_analysis(trend_data)
            
            content = template.format(
                metric=trend_data.metric_name,
                trend_direction=trend_data.trend_direction,
                change=abs(trend_data.change_percent),
                timeframe="24 hours",
                analysis=analysis,
                trend_description=f"{'positive' if trend_data.change_percent > 0 else 'negative'} trend",
                impact_description=f"{'improves' if trend_data.change_percent > 0 else 'impacts'} our market making efficiency",
                dashboard_url=self.config.get("dashboard.public_url", ""),
                hashtags=hashtags
            )
            
            return await self._create_post(content, PostType.TREND_ANALYSIS)
            
        except Exception as e:
            logger.error(f"Failed to post trend analysis: {e}")
            return None
    
    def _select_template(self, post_type: PostType) -> str:
        """Select random template for post type."""
        import random
        templates = self.post_templates.get(post_type, ["FlashMM update: {hashtags}"])
        return random.choice(templates)
    
    def _format_hashtags(self, categories: List[str]) -> str:
        """Format hashtags from categories."""
        all_tags = []
        for category in categories:
            if category in self.hashtags:
                all_tags.extend(self.hashtags[category])
        
        # Remove duplicates and limit to 8 hashtags
        unique_tags = list(dict.fromkeys(all_tags))[:8]
        return " ".join(unique_tags)
    
    def _generate_trend_analysis(self, trend_data: TrendData) -> str:
        """Generate trend analysis text."""
        if trend_data.change_percent > 10:
            return f"Strong {trend_data.trend_direction}ward movement indicates {trend_data.significance} market activity."
        elif trend_data.change_percent > 5:
            return f"Moderate {trend_data.trend_direction}ward trend with {trend_data.significance} impact on performance."
        elif trend_data.change_percent > 1:
            return f"Slight {trend_data.trend_direction}ward movement showing steady performance."
        else:
            return "Stable performance with minimal variation."
    
    async def _create_post(self, content: str, post_type: PostType) -> Optional[PostMetrics]:
        """Create a tweet/post."""
        try:
            if not self.posting_enabled or not self.session:
                logger.info(f"Skipping post (disabled): {content[:50]}...")
                return None
            
            # Check rate limits
            if not self._check_rate_limits():
                logger.warning("Rate limit exceeded, skipping post")
                return None
            
            # Ensure content fits Twitter limits
            content = self._truncate_content(content)
            
            # Create post data
            post_data = {
                "text": content
            }
            
            # Create OAuth headers
            headers = self._create_oauth_headers("POST", f"{self.base_url}/tweets")
            
            # Make API request
            async with self.session.post(
                f"{self.base_url}/tweets",
                json=post_data,
                headers=headers
            ) as response:
                if response.status in [200, 201]:
                    result = await response.json()
                    post_id = result.get("data", {}).get("id", "")
                    
                    # Create post metrics
                    post_metrics = PostMetrics(
                        post_id=post_id,
                        post_type=post_type,
                        content=content,
                        timestamp=datetime.now()
                    )
                    
                    # Track post
                    self.post_history.append(post_metrics)
                    self._update_rate_limits()
                    
                    logger.info(f"Successfully posted {post_type.value}: {post_id}")
                    return post_metrics
                    
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to create post: {error_text}")
                    return None
        
        except Exception as e:
            logger.error(f"Error creating post: {e}")
            return None
    
    def _can_post(self) -> bool:
        """Check if posting is enabled and allowed."""
        return (
            self.posting_enabled and 
            self.session is not None and
            self._check_rate_limits()
        )
    
    def _check_rate_limits(self) -> bool:
        """Check if within rate limits."""
        current_time = datetime.now()
        
        # Reset hourly counter
        if current_time.hour != self.rate_limits["last_reset_hour"]:
            self.rate_limits["hourly_posts"] = 0
            self.rate_limits["last_reset_hour"] = current_time.hour
        
        # Reset daily counter
        if current_time.date() != self.rate_limits["last_reset_date"]:
            self.rate_limits["daily_posts"] = 0
            self.rate_limits["last_reset_date"] = current_time.date()
        
        return (
            self.rate_limits["hourly_posts"] < self.rate_limits["posts_per_hour"] and
            self.rate_limits["daily_posts"] < self.rate_limits["posts_per_day"]
        )
    
    def _update_rate_limits(self) -> None:
        """Update rate limit counters."""
        self.rate_limits["hourly_posts"] += 1
        self.rate_limits["daily_posts"] += 1
        self.engagement_stats["total_posts"] += 1
    
    def _truncate_content(self, content: str, max_length: int = 280) -> str:
        """Truncate content to fit Twitter limits."""
        if len(content) <= max_length:
            return content
        
        # Find a good breaking point
        truncated = content[:max_length - 3]
        
        # Try to break at word boundary
        last_space = truncated.rfind(" ")
        if last_space > max_length * 0.8:  # If we can save at least 20% of content
            truncated = truncated[:last_space]
        
        return truncated + "..."
    
    async def get_post_analytics(self, post_id: str) -> Optional[PostMetrics]:
        """Get analytics for a specific post."""
        try:
            if not self.session:
                return None
            
            headers = self._create_oauth_headers("GET", f"{self.base_url}/tweets/{post_id}")
            params = {
                "tweet.fields": "public_metrics,created_at",
                "expansions": "author_id"
            }
            
            async with self.session.get(
                f"{self.base_url}/tweets/{post_id}",
                headers=headers,
                params=params
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    tweet_data = result.get("data", {})
                    metrics = tweet_data.get("public_metrics", {})
                    
                    # Find existing post metrics
                    for post_metrics in self.post_history:
                        if post_metrics.post_id == post_id:
                            # Update metrics
                            post_metrics.impressions = metrics.get("impression_count", 0)
                            post_metrics.likes = metrics.get("like_count", 0)
                            post_metrics.retweets = metrics.get("retweet_count", 0)
                            post_metrics.replies = metrics.get("reply_count", 0)
                            
                            # Calculate engagement rate
                            total_engagement = post_metrics.likes + post_metrics.retweets + post_metrics.replies
                            if post_metrics.impressions > 0:
                                post_metrics.engagement_rate = (total_engagement / post_metrics.impressions) * 100
                            
                            return post_metrics
                
                return None
        
        except Exception as e:
            logger.error(f"Error getting post analytics for {post_id}: {e}")
            return None
    
    async def update_all_analytics(self) -> Dict[str, Any]:
        """Update analytics for all recent posts."""
        try:
            # Update analytics for posts from last 7 days
            cutoff_date = datetime.now() - timedelta(days=7)
            recent_posts = [p for p in self.post_history if p.timestamp >= cutoff_date]
            
            updated_count = 0
            for post_metrics in recent_posts:
                if await self.get_post_analytics(post_metrics.post_id):
                    updated_count += 1
                    await asyncio.sleep(1)  # Rate limiting
            
            # Update engagement stats
            self._calculate_engagement_stats()
            
            logger.info(f"Updated analytics for {updated_count} posts")
            return self.engagement_stats
            
        except Exception as e:
            logger.error(f"Error updating analytics: {e}")
            return self.engagement_stats
    
    def _calculate_engagement_stats(self) -> None:
        """Calculate overall engagement statistics."""
        if not self.post_history:
            return
        
        total_impressions = sum(p.impressions for p in self.post_history)
        total_engagement = sum(p.likes + p.retweets + p.replies for p in self.post_history)
        
        self.engagement_stats.update({
            "total_posts": len(self.post_history),
            "total_impressions": total_impressions,
            "total_engagement": total_engagement,
            "avg_engagement_rate": (total_engagement / max(total_impressions, 1)) * 100,
            "best_performing_post": max(self.post_history, key=lambda p: p.engagement_rate, default=None)
        })
    
    async def schedule_regular_updates(self) -> None:
        """Schedule regular automated updates."""
        try:
            current_time = datetime.now()
            
            # Hourly performance updates (if configured)
            if (current_time.minute == 0 and 
                self.config.get("social.hourly_updates", False)):
                # Get latest metrics and post update
                # This would integrate with the metrics collector
                logger.info("Scheduling hourly update...")
            
            # Daily summary (posted at end of trading day)
            if (current_time.hour == 23 and current_time.minute == 0 and
                self.config.get("social.daily_summary", True)):
                logger.info("Scheduling daily summary...")
            
        except Exception as e:
            logger.error(f"Error scheduling updates: {e}")
    
    def get_engagement_summary(self) -> Dict[str, Any]:
        """Get comprehensive engagement summary."""
        return {
            "stats": self.engagement_stats,
            "recent_posts": [
                {
                    "id": p.post_id,
                    "type": p.post_type.value,
                    "content": p.content[:100] + "..." if len(p.content) > 100 else p.content,
                    "timestamp": p.timestamp.isoformat(),
                    "engagement_rate": p.engagement_rate,
                    "likes": p.likes,
                    "retweets": p.retweets
                }
                for p in sorted(self.post_history, key=lambda x: x.timestamp, reverse=True)[:10]
            ],
            "rate_limits": self.rate_limits
        }
    
    async def cleanup(self) -> None:
        """Cleanup Twitter client resources."""
        try:
            if self.session:
                await self.session.close()
                logger.info("TwitterClient session closed")
        except Exception as e:
            logger.error(f"Error during TwitterClient cleanup: {e}")


# Global Twitter client instance
_twitter_client: Optional[TwitterClient] = None


async def get_twitter_client() -> TwitterClient:
    """Get global Twitter client instance."""
    global _twitter_client
    if _twitter_client is None:
        _twitter_client = TwitterClient()
        await _twitter_client.initialize()
    return _twitter_client


async def cleanup_twitter_client() -> None:
    """Cleanup global Twitter client."""
    global _twitter_client
    if _twitter_client:
        await _twitter_client.cleanup()
        _twitter_client = None