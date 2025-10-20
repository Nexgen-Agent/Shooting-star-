# marketing/real_time_dashboard.py
class RealTimeMarketingDashboard:
    def __init__(self):
        self.data_streamer = DataStreamer()
        self.alert_engine = AlertEngine()
        
    async def get_real_time_metrics(self):
        """Get real-time marketing performance metrics"""
        return {
            "website_traffic": await self._get_live_traffic(),
            "social_engagement": await self._get_social_metrics(),
            "conversion_rates": await self._get_conversion_data(),
            "campaign_performance": await self._get_campaign_metrics()
        }
    
    async def generate_marketing_alerts(self):
        """Generate intelligent marketing alerts"""
        return await self._monitor_marketing_kpis()