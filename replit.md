# MEMO PRO

## Overview
MEMO PRO is designed to analyze written text using multi-model AI evaluation to assess authors' intelligence and cognitive fingerprints. It provides deep insights into cognitive abilities and thought processes from written content. Key capabilities include document analysis, AI detection, multi-language translation, comprehensive cognitive profiling, and intelligent text rewriting with advanced features for maximizing intelligence scores. Its business vision is to offer unparalleled textual analysis, catering to diverse market needs from academic research to professional content creation, aiming to become the leading platform for advanced cognitive text evaluation.

## User Preferences
Preferred communication style: Simple, everyday language.

## System Architecture
The application employs a monorepo structure, separating client and server components.

**UI/UX Decisions:**
- Frontend uses React with TypeScript, TailwindCSS, and shadcn/ui for a modern and responsive user interface.
- Data visualization is handled by Chart.js.
- Detailed card-based layouts are used for analysis reports.
- Supports PDF/text downloads, document upload, and output downloads.

**Technical Implementations & Feature Specifications:**
- **Frontend**: React, TypeScript, TailwindCSS, shadcn/ui, wouter, React Query, Chart.js.
- **Backend**: Express.js with TypeScript, integrating multiple LLMs, document processing, speech-to-text, and email services.
- **Database**: PostgreSQL with Drizzle ORM for user, document, analysis, and cognitive profile data.
- **Core Services**:
    - **Multi-Model Intelligence Evaluation**: A 4-phase system assessing 17 cognitive dimensions, supporting genre-aware analysis.
    - **Intelligent Rewrite Function (MAXINTEL)**: Recursively optimizes text for intelligence scores, with custom instructions and external knowledge integration.
    - **GPT Bypass Humanizer**: Transforms AI-generated text to bypass AI detection.
    - **Coherence Meter**: Supports up to 5000-word inputs with Global Coherence Preservation Protocol. Includes specialized modes:
        - **Mathematical Proof System** (COHERENCE, COGENCY, MAX COHERENCE, MAXIMIZE TRUTH).
        - **Scientific-Explanatory Coherence Type**: Dual assessment of logical consistency and scientific accuracy, with rewrite function to correct pseudoscientific claims.
    - **Text Model Validator**: Exclusively focused on the RECONSTRUCTION function for conservative charitable interpretation.
    - **AI Chat Assistant**: Provides conversation history and context from the Zhi Database.
    - **Conservative Reconstruction**: "Charitable Interpretation" mode for generating coherent essays articulating a text's unified argument, with advanced outline-first and cross-chunk strategies for medium and long documents.
        - **Short Input Support** (Dec 2025): Works with ANY input length, including 1-2 line pipe-delimited position lists
        - **Consolidated Output Mode**: Detects instructions like "WRITE ONE PARAGRAPH" or "EXPLAIN THESE STATEMENTS" and generates unified single-output responses instead of per-position defense blocks
        - **Position List Detection**: Threshold lowered from 10+ lines to 1+ lines for pipe-delimited format recognition
    - **Full Suite Pipeline**: One-click execution of Reconstruction, Objections, and Objection-Proof Final Version.
    - **Objections Function**: Generates 25 likely objections with compelling counter-arguments. For large documents (1,200+ words), uses outline-first approach that extracts argument structure first, then generates categorized objections (logical, evidential, practical, audience-specific, methodological) with severity ratings.
    - **Generate Objection-Proof Version (Bullet-Proof Rewrite)**: Rewrites text to preemptively address identified objections. Enhanced with:
        - **Claim-aware sectioning**: Detects claim-based structure (Claim 1:, Claim 2:, etc.) and preserves each claim with its paragraphs as a unit
        - **Header preservation**: Extracts and validates original claim headers, auto-prepends if missing
        - **Paragraph count enforcement**: Requires exact match, retries with stricter prompt if count differs
        - **Hedging detector**: Scans for 15 forbidden hedging phrases, retries if excessive hedging found
        - **Retry mechanism**: Up to 2 retries with progressively stricter instructions for structural compliance
        - **Anti-hedging guidance**: Produces confident prose that integrates objection-responses as assertions, not qualifications
        - **Two-Tier Format Preservation System** (Dec 2025):
            - **Custom Instruction Format Detection**: Detects glossary, numbered list, and non-paragraph formats from user instructions
            - **Input Format Detection**: Detects when reconstruction output is already in numbered format (e.g., "1. 'Claim...' Defense paragraphs")
            - **Format-Preserving Rewrite**: When numbered format detected in input, enforces EXACT item count and preserves quoted claims verbatim
            - **Direct Format Rewrite**: Bypasses section-based processing for special formats (glossaries, lists) to respect exact formatting requirements
    - **Recursive Objection-Proof Rewrite System** (Dec 2025):
        - **Infinite Iteration Support**: Allows unlimited rewrite iterations with custom instructions and target word count per iteration
        - **Cross-Chunk Coherence Architecture**: Uses GlobalSkeleton extraction, smart chunking, and constrained reconstruction for long documents
        - **Iteration Tracking**: Maintains iteration history with metadata (input/output word counts, timestamps, global skeleton)
        - **API Endpoints**:
            - `POST /api/objection-proof/iterations` - Start new recursive rewrite
            - `GET /api/objection-proof/iterations/:id` - Get iteration details
            - `POST /api/objection-proof/iterations/:id/refine` - Refine a previous iteration with new instructions
    - **Global Coherence State (GCS) System**: Architectural overhaul for coherence tracking across chunks, with mode-specific state dimensions for 8 coherence types.
    - **TextStats Component with AI Detection**: Displays word/character counts and GPTZero-powered AI detection results.

## External Dependencies
- **AI Service Providers**: OpenAI API (GPT-4), Anthropic API (Claude), DeepSeek API, Perplexity AI, Grok API (xAI).
- **Supporting Services**: Mathpix OCR, AssemblyAI, SendGrid, Google Custom Search, Stripe (for credit purchases), AnalyticPhilosophy.net Zhi API.
- **Database & Infrastructure**: Neon/PostgreSQL, Drizzle ORM, Replit.