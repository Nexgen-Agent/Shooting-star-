from typing import Dict, Any, List
from pathlib import Path
import json

class AdminPanelGenerator:
    """Generates custom admin panels for Brand Colonies"""
    
    def generate_admin_panel(self, brand_config: Dict, features: List[str]) -> Dict[str, str]:
        """Generate complete admin panel for brand colony"""
        
        admin_data = {
            "brand_name": brand_config["name"],
            "modules": self._generate_admin_modules(features),
            "dashboard": self._generate_dashboard_layout(features),
            "navigation": self._generate_navigation(features)
        }
        
        # Generate admin HTML structure
        admin_html = self._generate_admin_html(admin_data)
        
        # Generate admin CSS
        admin_css = self._generate_admin_css(brand_config)
        
        # Generate admin JavaScript
        admin_js = self._generate_admin_js(admin_data)
        
        return {
            "html": admin_html,
            "css": admin_css,
            "js": admin_js,
            "config": json.dumps(admin_data, indent=2)
        }
    
    def _generate_admin_modules(self, features: List[str]) -> List[Dict]:
        """Generate admin modules based on brand features"""
        modules = []
        
        module_map = {
            "ecommerce": [
                {"name": "Products", "icon": "📦", "endpoint": "/admin/products"},
                {"name": "Orders", "icon": "🛒", "endpoint": "/admin/orders"},
                {"name": "Inventory", "icon": "📊", "endpoint": "/admin/inventory"}
            ],
            "blog": [
                {"name": "Posts", "icon": "📝", "endpoint": "/admin/posts"},
                {"name": "Categories", "icon": "📑", "endpoint": "/admin/categories"},
                {"name": "Comments", "icon": "💬", "endpoint": "/admin/comments"}
            ],
            "analytics": [
                {"name": "Dashboard", "icon": "📈", "endpoint": "/admin/analytics"},
                {"name": "Reports", "icon": "📋", "endpoint": "/admin/reports"}
            ],
            "users": [
                {"name": "Customers", "icon": "👥", "endpoint": "/admin/customers"},
                {"name": "Team", "icon": "👨‍💼", "endpoint": "/admin/team"}
            ]
        }
        
        for feature in features:
            if feature in module_map:
                modules.extend(module_map[feature])
        
        return modules
    
    def _generate_dashboard_layout(self, features: List[str]) -> str:
        """Generate dashboard layout HTML"""
        widgets = []
        
        if "ecommerce" in features:
            widgets.append("""
            <div class="dashboard-widget sales-widget">
                <h3>Sales Overview</h3>
                <div class="widget-content">
                    <div class="metric">
                        <span class="value">$12,456</span>
                        <span class="label">Today's Revenue</span>
                    </div>
                    <div class="metric">
                        <span class="value">247</span>
                        <span class="label">New Orders</span>
                    </div>
                </div>
            </div>
            """)
        
        if "analytics" in features:
            widgets.append("""
            <div class="dashboard-widget analytics-widget">
                <h3>Website Analytics</h3>
                <div class="widget-content">
                    <div class="metric">
                        <span class="value">4,892</span>
                        <span class="label">Visitors Today</span>
                    </div>
                    <div class="metric">
                        <span class="value">2.3%</span>
                        <span class="label">Conversion Rate</span>
                    </div>
                </div>
            </div>
            """)
        
        if not widgets:
            widgets.append("""
            <div class="dashboard-widget welcome-widget">
                <h3>Welcome to Your Admin Panel</h3>
                <p>Your business dashboard will appear here once you start getting data.</p>
            </div>
            """)
        
        return '\n'.join(widgets)
    
    def _generate_navigation(self, features: List[str]) -> str:
        """Generate navigation menu HTML"""
        nav_items = ['<li><a href="/admin/dashboard">Dashboard</a></li>']
        
        if "ecommerce" in features:
            nav_items.extend([
                '<li><a href="/admin/products">Products</a></li>',
                '<li><a href="/admin/orders">Orders</a></li>'
            ])
        
        if "blog" in features:
            nav_items.extend([
                '<li><a href="/admin/posts">Blog Posts</a></li>',
                '<li><a href="/admin/categories">Categories</a></li>'
            ])
        
        nav_items.append('<li><a href="/admin/settings">Settings</a></li>')
        
        return '\n'.join(nav_items)
    
    def _generate_admin_html(self, admin_data: Dict) -> str:
        """Generate complete admin panel HTML"""
        return f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{admin_data['brand_name']} - Admin Panel</title>
            <link rel="stylesheet" href="assets/css/admin.css">
        </head>
        <body>
            <div class="admin-container">
                <header class="admin-header">
                    <div class="branding">
                        <h1>{admin_data['brand_name']} Admin</h1>
                    </div>
                    <nav class="user-menu">
                        <span>Welcome, Admin</span>
                        <a href="/logout">Logout</a>
                    </nav>
                </header>
                
                <div class="admin-layout">
                    <aside class="sidebar">
                        <nav class="main-nav">
                            <ul>
                                {admin_data['navigation']}
                            </ul>
                        </nav>
                    </aside>
                    
                    <main class="admin-main">
                        <div class="dashboard-grid">
                            {admin_data['dashboard']}
                        </div>
                    </main>
                </div>
            </div>
            
            <script src="assets/js/admin.js"></script>
        </body>
        </html>
        """
    
    def _generate_admin_css(self, brand_config: Dict) -> str:
        """Generate admin panel CSS"""
        primary_color = brand_config.get('primary_color', '#007bff')
        
        return f"""
        /* Admin Panel Styles for {brand_config['name']} */
        :root {{
            --primary-color: {primary_color};
            --sidebar-width: 250px;
        }}
        
        .admin-container {{
            display: flex;
            flex-direction: column;
            height: 100vh;
        }}
        
        .admin-header {{
            background: #2c3e50;
            color: white;
            padding: 1rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .admin-layout {{
            display: flex;
            flex: 1;
        }}
        
        .sidebar {{
            width: var(--sidebar-width);
            background: #34495e;
            color: white;
        }}
        
        .main-nav ul {{
            list-style: none;
            padding: 0;
        }}
        
        .main-nav a {{
            color: white;
            text-decoration: none;
            padding: 0.75rem 1rem;
            display: block;
            border-bottom: 1px solid #2c3e50;
        }}
        
        .main-nav a:hover {{
            background: var(--primary-color);
        }}
        
        .admin-main {{
            flex: 1;
            padding: 2rem;
            background: #ecf0f1;
        }}
        
        .dashboard-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
        }}
        
        .dashboard-widget {{
            background: white;
            border-radius: 8px;
            padding: 1.5rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        .dashboard-widget h3 {{
            margin-top: 0;
            color: #2c3e50;
        }}
        """
    
    def _generate_admin_js(self, admin_data: Dict) -> str:
        """Generate admin panel JavaScript"""
        return f"""
        // Admin Panel JavaScript for {admin_data['brand_name']}
        class AdminPanel {{
            constructor() {{
                this.brandName = "{admin_data['brand_name']}";
                this.init();
            }}
            
            init() {{
                this.loadDashboardData();
                this.setupEventListeners();
                console.log('Admin panel initialized for', this.brandName);
            }}
            
            loadDashboardData() {{
                // Load initial dashboard data
                fetch('/api/admin/dashboard')
                    .then(response => response.json())
                    .then(data => this.updateDashboard(data));
            }}
            
            updateDashboard(data) {{
                // Update dashboard widgets with real data
                console.log('Dashboard updated:', data);
            }}
            
            setupEventListeners() {{
                // Setup navigation and interactions
                document.addEventListener('click', this.handleClick.bind(this));
            }}
            
            handleClick(event) {{
                // Handle admin panel interactions
                if (event.target.matches('.nav-link')) {{
                    this.navigateTo(event.target.href);
                }}
            }}
            
            navigateTo(url) {{
                // Handle navigation
                window.location.href = url;
            }}
        }}
        
        // Initialize admin panel when DOM is loaded
        document.addEventListener('DOMContentLoaded', () => {{
            new AdminPanel();
        }});
        """