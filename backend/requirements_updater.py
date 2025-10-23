"""
Enhanced requirements for the V16 AI system with all dependencies.
"""

REQUIREMENTS_V16 = {
    "core": {
        "fastapi": ">=0.104.0,<0.105.0",
        "uvicorn": ">=0.24.0,<0.25.0",
        "pydantic": ">=2.0.0,<3.0.0",
        "python-multipart": ">=0.0.6,<0.1.0"
    },
    "ai_ml": {
        "numpy": ">=1.24.0,<1.25.0",
        "pandas": ">=2.0.0,<3.0.0",
        "scikit-learn": ">=1.3.0,<1.4.0",
        "torch": ">=2.0.0,<2.1.0",
        "transformers": ">=4.35.0,<4.36.0",
        "sentence-transformers": ">=2.2.0,<2.3.0",
        "langchain": ">=0.0.350,<0.1.0",
        "openai": ">=1.0.0,<2.0.0",
        "prophet": ">=1.1.0,<1.2.0"
    },
    "database": {
        "sqlalchemy": ">=2.0.0,<2.1.0",
        "asyncpg": ">=0.28.0,<0.29.0",
        "redis": ">=5.0.0,<5.1.0",
        "influxdb-client": ">=1.36.0,<1.37.0"
    },
    "audio_processing": {
        "speechrecognition": ">=3.10.0,<3.11.0",
        "pyaudio": ">=0.2.11,<0.3.0",
        "gtts": ">=2.3.0,<2.4.0"
    },
    "image_processing": {
        "pillow": ">=10.0.0,<10.1.0",
        "opencv-python": ">=4.8.0,<4.9.0"
    },
    "monitoring": {
        "sentry-sdk": ">=1.40.0,<1.41.0",
        "prometheus-client": ">=0.19.0,<0.20.0",
        "datadog": ">=0.48.0,<0.49.0"
    },
    "utilities": {
        "celery": ">=5.3.0,<5.4.0",
        "apscheduler": ">=3.10.0,<3.11.0",
        "python-jose": ">=3.3.0,<3.4.0",
        "passlib": ">=1.7.4,<1.8.0",
        "bcrypt": ">=4.0.0,<4.1.0"
    }
}

def generate_requirements_file():
    """Generate requirements.txt file for V16 system"""
    requirements = []
    
    for category, packages in REQUIREMENTS_V16.items():
        requirements.append(f"# {category.upper()} dependencies")
        for package, version in packages.items():
            requirements.append(f"{package}{version}")
        requirements.append("")
    
    requirements_content = "\n".join(requirements)
    
    with open("requirements_v16.txt", "w") as f:
        f.write(requirements_content)
    
    print("✅ requirements_v16.txt generated successfully")
    return requirements_content

# Additional development dependencies
DEV_REQUIREMENTS = {
    "testing": {
        "pytest": ">=7.4.0,<7.5.0",
        "pytest-asyncio": ">=0.21.0,<0.22.0",
        "pytest-cov": ">=4.1.0,<4.2.0",
        "httpx": ">=0.25.0,<0.26.0"
    },
    "development": {
        "black": ">=23.0.0,<24.0.0",
        "flake8": ">=6.1.0,<6.2.0",
        "mypy": ">=1.7.0,<1.8.0",
        "pre-commit": ">=3.5.0,<3.6.0"
    },
    "documentation": {
        "mkdocs": ">=1.5.0,<1.6.0",
        "mkdocs-material": ">=9.4.0,<9.5.0",
        "pydocstyle": ">=6.3.0,<6.4.0"
    }
}

if __name__ == "__main__":
    generate_requirements_file()