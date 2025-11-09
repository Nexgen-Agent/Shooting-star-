import os
import shutil
import json
from typing import Dict, Any, List
from pathlib import Path
from jinja2 import Template
import logging

logger = logging.getLogger(__name__)

class WebsiteGenerator:
    """AI-Powered Website Generator for Brand Colonies"""
    
    def __init__(self):
        self.template_dir = Path("frontend/templates")
        self.build_dir = Path("frontend/build")
        self.available_templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, Any]:
        """Load available website templates"""
        return {
            "ecommerce": {
                "name": "E-Commerce Store",
                "components": ["product_catalog", "shopping_cart", "checkout", "inventory"],
                "styles": ["modern", "minimal", "corporate"],
                "pages": ["home", "products", "about", "contact"]
            },
            "saas": {
                "name": "SaaS Platform", 
                "components": ["pricing", "features", "dashboard", "auth"],
                "styles": ["tech", "clean", "gradient"],
                "pages": ["home", "pricing", "features", "login"]
            },
            "agency": {
                "name": "Agency Portfolio",
                "components": ["portfolio", "services", "testimonials", "contact_form"],
                "styles": ["creative", "bold", "elegant"],
                "pages": ["home", "services", "work", "about", "contact"]
            },
            "restaurant": {
                "name": "Restaurant Website",
                "components": ["menu", "reservations", "gallery", "location"],
                "styles": ["warm", "modern", "rustic"],
                "pages": ["home", "menu", "about", "reservations", "contact"]
            }
        }
    
    def generate_brand_website(self, brand_config: Dict[str, Any]) -> Dict[str, str]:
        """Generate complete website for a brand colony"""
        try:
            template_type = brand_config.get("industry", "agency")
            brand_name = brand_config["name"]
            brand_slug = self._slugify(brand_name)
            
            # Create brand directory
            brand_build_path = self.build_dir / brand_slug
            brand_build_path.mkdir(parents=True, exist_ok=True)
            
            # Generate website structure
            website_data = {
                "brand": brand_config,
                "generated_at": datetime.utcnow().isoformat(),
                "template_used": template_type
            }
            
            # Generate HTML pages
            self._generate_html_pages(brand_build_path, brand_config, template_type)
            
            # Generate CSS styles
            self._generate_css_styles(brand_build_path, brand_config, template_type)
            
            # Generate JavaScript functionality
            self._generate_javascript(brand_build_path, brand_config)
            
            # Generate admin panel
            admin_panel_url = self._generate_admin_panel(brand_build_path, brand_config)
            
            # Deploy website (simulated)
            website_url = self._deploy_website(brand_build_path, brand_slug)
            
            logger.info(f"Successfully generated website for {brand_name}")
            
            return {
                "website_url": website_url,
                "admin_panel_url": admin_panel_url,
                "build_path": str(brand_build_path),
                "brand_slug": brand_slug
            }
            
        except Exception as e:
            logger.error(f"Failed to generate website: {str(e)}")
            raise
    
    def _generate_html_pages(self, build_path: Path, brand_config: Dict, template_type: str):
        """Generate HTML pages for the website"""
        template_config = self.available_templates[template_type]
        
        for page in template_config["pages"]:
            html_content = self._render_template(
                f"{template_type}/{page}.html.j2", 
                brand_config
            )
            
            page_file = build_path / f"{page}.html"
            if page == "home":
                page_file = build_path / "index.html"
                
            page_file.write_text(html_content)
    
    def _generate_css_styles(self, build_path: Path, brand_config: Dict, template_type: str):
        """Generate custom CSS styles"""
        css_content = self._render_template(
            f"{template_type}/style.css.j2",
            {**brand_config, "template_type": template_type}
        )
        
        css_file = build_path / "assets" / "css" / "style.css"
        css_file.parent.mkdir(parents=True, exist_ok=True)
        css_file.write_text(css_content)
    
    def _generate_javascript(self, build_path: Path, brand_config: Dict):
        """Generate JavaScript functionality"""
        js_content = f"""
        // Auto-generated JavaScript for {brand_config['name']}
        document.addEventListener('DOMContentLoaded', function() {{
            console.log('{brand_config['name']} website loaded');
            
            // Initialize brand-specific functionality
            initBrandWebsite({json.dumps(brand_config)});
        }});
        
        function initBrandWebsite(config) {{
            // Dynamic content loading
            // Form handling
            // Interactive elements
            console.log('Initialized:', config.name);
        }}
        """
        
        js_file = build_path / "assets" / "js" / "main.js"
        js_file.parent.mkdir(parents=True, exist_ok=True)
        js_file.write_text(js_content)
    
    def _generate_admin_panel(self, build_path: Path, brand_config: Dict) -> str:
        """Generate admin panel for brand colony"""
        admin_path = build_path / "admin"
        admin_path.mkdir(exist_ok=True)
        
        # Generate admin HTML
        admin_html = self._render_template("admin/panel.html.j2", brand_config)
        (admin_path / "index.html").write_text(admin_html)
        
        # Generate admin CSS
        admin_css = self._render_template("admin/style.css.j2", brand_config)
        (admin_path / "assets" / "css" / "admin.css").parent.mkdir(parents=True, exist_ok=True)
        (admin_path / "assets" / "css" / "admin.css").write_text(admin_css)
        
        # Generate admin JavaScript
        admin_js = f"""
        // Admin Panel for {brand_config['name']}
        class BrandAdminPanel {{
            constructor(config) {{
                this.brandName = config.name;
                this.init();
            }}
            
            init() {{
                this.loadDashboard();
                this.setupEventListeners();
            }}
            
            loadDashboard() {{
                // Load orders, analytics, etc.
                console.log('Admin panel loaded for', this.brandName);
            }}
        }}
        
        new BrandAdminPanel({json.dumps(brand_config)});
        """
        (admin_path / "assets" / "js" / "admin.js").parent.mkdir(parents=True, exist_ok=True)
        (admin_path / "assets" / "js" / "admin.js").write_text(admin_js)
        
        return f"/admin/{brand_config['slug']}/admin"
    
    def _render_template(self, template_path: str, context: Dict) -> str:
        """Render Jinja2 template"""
        template_file = self.template_dir / template_path
        if template_file.exists():
            template_content = template_file.read_text()
            template = Template(template_content)
            return template.render(**context)
        else:
            # Return basic template if specific one doesn't exist
            return self._get_fallback_template(template_path, context)
    
    def _get_fallback_template(self, template_type: str, context: Dict) -> str:
        """Provide fallback templates"""
        if template_type.endswith(".html.j2"):
            return f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>{context.get('name', 'Brand Website')}</title>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
            </head>
            <body>
                <header>
                    <h1>{context.get('name', 'Welcome')}</h1>
                    <nav>
                        <a href="/">Home</a>
                        <a href="/about">About</a>
                        <a href="/contact">Contact</a>
                    </nav>
                </header>
                <main>
                    <h2>Welcome to {context.get('name', 'Our Business')}</h2>
                    <p>This website was automatically generated by The Colony AI System.</p>
                </main>
                <footer>
                    <p>&copy; 2024 {context.get('name', 'Brand')}. All rights reserved.</p>
                </footer>
            </body>
            </html>
            """
        return ""
    
    def _slugify(self, text: str) -> str:
        """Convert text to URL-friendly slug"""
        return text.lower().replace(' ', '-').replace('--', '-')
    
    def _deploy_website(self, build_path: Path, brand_slug: str) -> str:
        """Deploy website to hosting (simulated)"""
        # In production, this would deploy to Netlify, Vercel, S3, etc.
        website_url = f"https://{brand_slug}.colony-websites.com"
        logger.info(f"Deployed website to: {website_url}")
        return website_url