from setuptools import setup, find_packages

setup(
    name="ai-note-taking-assistant",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "python-docx",
        "PyPDF2",
        "selenium",
        "python-magic",
        "langchain",
        "openai",
        "chromadb",
        "python-dotenv",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="An AI-powered note-taking assistant for processing documents and videos",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ai-note-taking-assistant",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
) 