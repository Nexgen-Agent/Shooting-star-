# extensions/vbe/cheese_method.py
"""
Cheese Method Outreach System
Advanced outreach template system following the Cheese Method framework
"""
from typing import Dict, List, Optional
import random


def build_outreach_message(profile: dict, service: str, tone: str = "confident") -> dict:
    """
    Build outreach message using Cheese Method template system
    
    Args:
        profile: Lead profile with name, org, title, etc.
        service: Service to offer from business services list
        tone: Message tone - "confident", "humble", or "playful"
    
    Returns:
        dict: Complete outreach message with subject, body, and metadata
        
    Example:
        >>> profile = {"name": "John", "org": "TechCo", "title": "CEO"}
        >>> message = build_outreach_message(profile, "website builds", "confident")
        >>> "subject" in message
        True
    """
    
    # Hook: Personalized opener based on public detail
    hook = _generate_hook(profile, tone)
    
    # Value: Clear value proposition
    value_prop = _generate_value_proposition(service, tone)
    
    # Social Proof: Case study or result
    social_proof = _generate_social_proof(service)
    
    # Offer: Low-friction next step
    offer = _generate_offer(service, tone)
    
    # Scarcity/CTA: Time-limited invite
    cta = _generate_cta(tone)
    
    # Build complete message
    body = f"{hook}\n\n{value_prop}\n\n{social_proof}\n\n{offer}\n\n{cta}"
    
    subject = generate_subject(service, tone)
    
    return {
        "subject": subject,
        "body": body,
        "preview_snippets": [hook, value_prop, social_proof],
        "tags": [service, tone, "cheese_method"],
        "html_body": format_for_email(body, as_html=True),
        "plain_text": format_for_email(body, as_html=False)
    }


def generate_subject(service: str, tone: str = "confident") -> str:
    """
    Generate email subject line based on service and tone
    
    Args:
        service: Service being offered
        tone: Tone variant
        
    Returns:
        str: Generated subject line
    """
    service_subjects = {
        "website builds": [
            "Quick question about your website",
            "Ideas for your online presence",
            "Noticed your amazing work"
        ],
        "digital marketing": [
            "Growth opportunity for your business", 
            "Digital strategy insights",
            "Marketing performance boost"
        ],
        "creator growth services": [
            "Content strategy ideas for you",
            "Audience growth opportunity",
            "Creator partnership idea"
        ]
    }
    
    tone_modifiers = {
        "confident": ["Proven approach", "Results guaranteed", "Expert insight"],
        "humble": ["Humble suggestion", "If I may", "Quick thought"],
        "playful": ["Fun idea", "Creative approach", "Exciting opportunity"]
    }
    
    base_subjects = service_subjects.get(service, ["Opportunity to connect"])
    modifiers = tone_modifiers.get(tone, [""])
    
    subject = f"{random.choice(modifiers)}: {random.choice(base_subjects)}"
    return subject.strip()


def generate_body_variant(hook: str, value_prop: str, social_proof: str, 
                         offer: str, cta: str, variant: str = "standard") -> str:
    """
    Generate body text variant with different formatting
    
    Args:
        hook: Opening hook
        value_prop: Value proposition
        social_proof: Social proof element
        offer: Service offer
        cta: Call to action
        variant: Format variant
        
    Returns:
        str: Formatted body text
    """
    variants = {
        "standard": f"{hook}\n\n{value_prop}\n\n{social_proof}\n\n{offer}\n\n{cta}",
        "concise": f"{hook}\n{value_prop}\n{offer}\n{cta}",
        "detailed": f"{hook}\n\n{value_prop}\n\n{social_proof}\n\n{offer}\n\nAdditional context...\n\n{cta}"
    }
    
    return variants.get(variant, variants["standard"])


def format_for_email(body: str, as_html: bool = False) -> str:
    """
    Format body text for email (HTML or plain text)
    
    Args:
        body: Raw body text
        as_html: Whether to format as HTML
        
    Returns:
        str: Formatted email body
    """
    if not as_html:
        return body
    
    # Convert to simple HTML formatting
    paragraphs = body.split('\n\n')
    html_paragraphs = []
    
    for p in paragraphs:
        if p.strip():
            html_paragraphs.append(f"<p>{p.strip()}</p>")
    
    return "\n".join(html_paragraphs)


def _generate_hook(profile: dict, tone: str) -> str:
    """Generate personalized hook based on lead profile"""
    name = profile.get("name", "there")
    org = profile.get("org", "your company")
    
    hooks = {
        "confident": [
            f"Loved what {org} is doing in the space - particularly impressed with your recent initiatives.",
            f"Noticed {org}'s growth trajectory and wanted to reach out about a strategic opportunity.",
            f"Your work at {org} caught my attention - especially the innovative approach you're taking."
        ],
        "humble": [
            f"Really admire what you're building at {org} and wanted to respectfully share an idea.",
            f"Hope you don't mind me reaching out - been following {org}'s progress and am impressed.",
            f"If I may, I wanted to share a thought about {org}'s current positioning."
        ],
        "playful": [
            f"Okay, I'll cut right to it - what {org} is doing is seriously cool!",
            f"Have to say, I'm a big fan of how {org} is shaking things up!",
            f"Quick question that might be fun to explore together regarding {org}'s next phase."
        ]
    }
    
    return random.choice(hooks.get(tone, hooks["confident"]))


def _generate_value_proposition(service: str, tone: str) -> str:
    """Generate value proposition for specific service"""
    value_props = {
        "website builds": "We build high-converting websites that drive 3x more qualified leads within 30 days.",
        "digital marketing": "Our data-driven marketing strategies typically increase ROAS by 200-400% in first quarter.",
        "creator growth services": "We help creators systematize content and grow audience 47% faster while saving 15+ hours weekly.",
        "graphic design": "Professional design assets that increase brand recognition and conversion rates by up to 80%.",
        "branding & positioning": "Strategic positioning that clarifies your message and attracts ideal customers automatically.",
        "cybersecurity & maintenance": "Enterprise-grade security and maintenance ensuring 99.9% uptime and zero vulnerabilities."
    }
    
    base_prop = value_props.get(service, "Strategic solutions that drive measurable growth and efficiency.")
    
    tone_wrappers = {
        "confident": f"I'm confident we can deliver: {base_prop}",
        "humble": f"Based on our experience, we typically see results like: {base_prop}",
        "playful": f"Here's the exciting part: {base_prop}"
    }
    
    return tone_wrappers.get(tone, base_prop)


def _generate_social_proof(service: str) -> str:
    """Generate social proof case study"""
    case_studies = {
        "website builds": "Recent client saw 214% increase in lead conversion after website redesign.",
        "digital marketing": "E-commerce client achieved 327% ROAS and $2.3M incremental revenue in 6 months.",
        "creator growth services": "Creator grew from 10K to 85K followers in 4 months while doubling engagement rate.",
        "subscription products": "Client scaled from $15K to $125K MRR in 9 months with our growth framework."
    }
    
    return case_studies.get(service, "Our clients typically see 2-3x improvement in key metrics within first 90 days.")


def _generate_offer(service: str, tone: str) -> str:
    """Generate low-friction offer"""
    offers = {
        "confident": f"Happy to do a quick 15-minute audit of your current {service} setup and share specific ideas.",
        "humble": f"If it's helpful, I'd be glad to offer a complimentary 15-minute consultation on {service}.",
        "playful": f"Want to explore this? I'll gift you a 15-minute {service} strategy session - no strings attached."
    }
    
    return offers.get(tone, f"Would you be open to a quick 15-minute chat about {service}?")


def _generate_cta(tone: str) -> str:
    """Generate call to action with scarcity"""
    ctas = {
        "confident": "I have 2 slots open next week for complementary audits - let me know if Tuesday or Wednesday works.",
        "humble": "I have limited capacity this month but would be honored to collaborate if timing aligns.",
        "playful": "Quick heads-up - I only have 3 complimentary session slots this month. Want to grab one?"
    }
    
    return ctas.get(tone, "I have limited availability next week - would you like to schedule a quick call?")


if __name__ == "__main__":
    # Debug harness
    test_profile = {
        "name": "Sarah",
        "org": "InnovateTech", 
        "title": "Marketing Director",
        "profile_url": "https://linkedin.com/in/sarah-tech"
    }
    
    message = build_outreach_message(test_profile, "digital marketing", "confident")
    print("=== CHEESE METHOD OUTREACH ===")
    print(f"Subject: {message['subject']}")
    print(f"Body:\n{message['body']}")
    print(f"Tags: {message['tags']}")