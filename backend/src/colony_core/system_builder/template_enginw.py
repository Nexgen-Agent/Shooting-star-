from typing import Dict, Any
from pathlib import Path
import json

class TemplateEngine:
    """Manages website templates and themes"""
    
    def __init__(self):
        self.templates_path = Path("frontend/templates")
        self.component_library = self._load_component_library()
    
    def _load_component_library(self) -> Dict[str, Any]:
        """Load reusable UI components"""
        return {
            "headers": {
                "basic": """
                <header class="brand-header">
                    <div class="container">
                        <div class="logo">
                            <h1>{{ brand.name }}</h1>
                        </div>
                        <nav class="main-nav">
                            {% for page in brand.pages %}
                            <a href="/{{ page.slug }}">{{ page.name }}</a>
                            {% endfor %}
                        </nav>
                    </div>
                </header>
                """,
                "modern": """
                <header class="modern-header">
                    <div class="container">
                        <div class="branding">
                            <h1 class="logo">{{ brand.name }}</h1>
                            <p class="tagline">{{ brand.tagline }}</p>
                        </div>
                        <nav class="navigation">
                            {% for page in brand.pages %}
                            <a href="/{{ page.slug }}" class="nav-link">{{ page.name }}</a>
                            {% endfor %}
                        </nav>
                    </div>
                </header>
                """
            },
            "footers": {
                "basic": """
                <footer class="brand-footer">
                    <div class="container">
                        <p>&copy; 2024 {{ brand.name }}. All rights reserved.</p>
                        <div class="social-links">
                            {% if brand.social_media %}
                            {% for platform, url in brand.social_media.items() %}
                            <a href="{{ url }}">{{ platform|title }}</a>
                            {% endfor %}
                            {% endif %}
                        </div>
                    </div>
                </footer>
                """
            },
            "hero_sections": {
                "business": """
                <section class="hero business-hero">
                    <div class="container">
                        <h1>{{ brand.hero_title or "Welcome to " + brand.name }}</h1>
                        <p>{{ brand.hero_subtitle or "Your success is our mission" }}</p>
                        <div class="cta-buttons">
                            <a href="/contact" class="btn btn-primary">Get Started</a>
                            <a href="/about" class="btn btn-secondary">Learn More</a>
                        </div>
                    </div>
                </section>
                """,
                "ecommerce": """
                <section class="hero ecommerce-hero">
                    <div class="container">
                        <h1>Discover Amazing Products</h1>
                        <p>Shop the latest collection from {{ brand.name }}</p>
                        <a href="/products" class="btn btn-shop">Shop Now</a>
                    </div>
                </section>
                """
            }
        }
    
    def generate_component(self, component_type: str, style: str, brand_data: Dict) -> str:
        """Generate a specific UI component"""
        from jinja2 import Template
        
        if component_type in self.component_library and style in self.component_library[component_type]:
            template_str = self.component_library[component_type][style]
            template = Template(template_str)
            return template.render(brand=brand_data)
        
        return f"<!-- Component {component_type}.{style} not found -->"
    
    def create_custom_template(self, brand_requirements: Dict) -> Dict[str, str]:
        """Create custom template based on brand requirements"""
        template_data = {
            "layout": "responsive",
            "color_scheme": brand_requirements.get("colors", {"primary": "#007bff"}),
            "typography": brand_requirements.get("fonts", {"primary": "Arial, sans-serif"}),
            "components": []
        }
        
        # Generate component list based on requirements
        if "ecommerce" in brand_requirements.get("features", []):
            template_data["components"].extend(["product_grid", "shopping_cart", "checkout_form"])
        
        if "blog" in brand_requirements.get("features", []):
            template_data["components"].extend(["blog_list", "post_detail", "comments"])
        
        return template_data