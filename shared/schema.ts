import { pgTable, text, serial, integer, boolean, jsonb, timestamp, real } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

// Updated User model - username/password authentication
export const users = pgTable("users", {
  id: serial("id").primaryKey(),
  username: text("username").notNull().unique(),
  password: text("password").notNull(),
  email: text("email"), // Optional email field
  createdAt: timestamp("created_at").defaultNow().notNull(),
  lastActiveAt: timestamp("last_active_at").defaultNow().notNull(),
});

export const insertUserSchema = createInsertSchema(users).pick({
  username: true,
  password: true,
  email: true,
});

// Document model for storing analyzed documents
export const documents = pgTable("documents", {
  id: serial("id").primaryKey(),
  content: text("content").notNull(),
  originalContent: text("original_content"), // Store original content
  filename: text("filename"),
  mimeType: text("mime_type"),
  userId: integer("user_id").references(() => users.id),
  wordCount: integer("word_count"),
  mathNotationCount: integer("math_notation_count"), // Count of LaTeX expressions
  complexity: text("complexity"), // low, medium, high
  createdAt: timestamp("created_at").defaultNow().notNull(),
  aiProbability: integer("ai_probability"),
  isAi: boolean("is_ai"),
});

export const insertDocumentSchema = createInsertSchema(documents).pick({
  content: true,
  originalContent: true,
  filename: true,
  mimeType: true,
  userId: true,
  wordCount: true,
  mathNotationCount: true,
  complexity: true,
});

// Analysis model for storing document analysis results
export const analyses = pgTable("analyses", {
  id: serial("id").primaryKey(),
  documentId: integer("document_id").references(() => documents.id).notNull(),
  userId: integer("user_id").references(() => users.id),
  summary: text("summary").notNull(),
  overallScore: integer("overall_score").notNull(),
  overallAssessment: text("overall_assessment").notNull(),
  dimensions: jsonb("dimensions").notNull(),
  cognitivePatterns: jsonb("cognitive_patterns"), // AI-identified thinking patterns
  writingStyle: jsonb("writing_style"), // Style analysis
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

export const insertAnalysisSchema = createInsertSchema(analyses).pick({
  documentId: true,
  userId: true,
  summary: true,
  overallScore: true,
  overallAssessment: true,
  dimensions: true,
  cognitivePatterns: true,
  writingStyle: true,
});

// Intelligent Rewrite model for storing rewrite results
export const intelligentRewrites = pgTable("intelligent_rewrites", {
  id: serial("id").primaryKey(),
  originalDocumentId: integer("original_document_id").references(() => documents.id).notNull(),
  rewrittenDocumentId: integer("rewritten_document_id").references(() => documents.id).notNull(),
  originalAnalysisId: integer("original_analysis_id").references(() => analyses.id).notNull(),
  rewrittenAnalysisId: integer("rewritten_analysis_id").references(() => analyses.id).notNull(),
  userId: integer("user_id").references(() => users.id),
  provider: text("provider").notNull(),
  customInstructions: text("custom_instructions"),
  originalScore: integer("original_score").notNull(),
  rewrittenScore: integer("rewritten_score").notNull(),
  scoreImprovement: integer("score_improvement").notNull(),
  rewriteReport: text("rewrite_report").notNull(),
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

export const insertIntelligentRewriteSchema = createInsertSchema(intelligentRewrites).pick({
  originalDocumentId: true,
  rewrittenDocumentId: true,
  originalAnalysisId: true,
  rewrittenAnalysisId: true,
  userId: true,
  provider: true,
  customInstructions: true,
  originalScore: true,
  rewrittenScore: true,
  scoreImprovement: true,
  rewriteReport: true,
});

// Comparison model for storing comparison results between two documents
export const comparisons = pgTable("comparisons", {
  id: serial("id").primaryKey(),
  documentAId: integer("document_a_id").references(() => documents.id).notNull(),
  documentBId: integer("document_b_id").references(() => documents.id).notNull(),
  analysisAId: integer("analysis_a_id").references(() => analyses.id).notNull(),
  analysisBId: integer("analysis_b_id").references(() => analyses.id).notNull(),
  userId: integer("user_id").references(() => users.id),
  comparisonResults: jsonb("comparison_results").notNull(),
  improvementSuggestions: jsonb("improvement_suggestions"),
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

export const insertComparisonSchema = createInsertSchema(comparisons).pick({
  documentAId: true,
  documentBId: true,
  analysisAId: true,
  analysisBId: true,
  userId: true,
  comparisonResults: true,
  improvementSuggestions: true,
});

// Case assessment model for evaluating how well documents make their case
export const caseAssessments = pgTable("case_assessments", {
  id: serial("id").primaryKey(),
  documentId: integer("document_id").references(() => documents.id).notNull(),
  userId: integer("user_id").references(() => users.id),
  proofEffectiveness: integer("proof_effectiveness").notNull(), // 0-100 score
  claimCredibility: integer("claim_credibility").notNull(), // 0-100 score  
  nonTriviality: integer("non_triviality").notNull(), // 0-100 score
  proofQuality: integer("proof_quality").notNull(), // 0-100 score
  functionalWriting: integer("functional_writing").notNull(), // 0-100 score
  overallCaseScore: integer("overall_case_score").notNull(), // 0-100 score
  detailedAssessment: text("detailed_assessment").notNull(), // Full LLM-generated report
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

export const insertCaseAssessmentSchema = createInsertSchema(caseAssessments).pick({
  documentId: true,
  userId: true,
  proofEffectiveness: true,
  claimCredibility: true,
  nonTriviality: true,
  proofQuality: true,
  functionalWriting: true,
  overallCaseScore: true,
  detailedAssessment: true,
});

// User activity tracking for cognitive pattern analysis
export const userActivities = pgTable("user_activities", {
  id: serial("id").primaryKey(),
  userId: integer("user_id").references(() => users.id),
  activityType: text("activity_type").notNull(), // upload, analyze, compare, search
  activityData: jsonb("activity_data"), // Detailed activity information
  documentId: integer("document_id").references(() => documents.id),
  sessionDuration: integer("session_duration"), // in seconds
  timestamp: timestamp("timestamp").defaultNow().notNull(),
});

export const insertUserActivitySchema = createInsertSchema(userActivities).pick({
  userId: true,
  activityType: true,
  activityData: true,
  documentId: true,
  sessionDuration: true,
});

// Comprehensive cognitive profiles - the core analytics system
export const cognitiveProfiles = pgTable("cognitive_profiles", {
  id: serial("id").primaryKey(),
  userId: integer("user_id").references(() => users.id).unique(),
  // Writing patterns and intellectual analysis
  writingPatterns: jsonb("writing_patterns"), // Sentence structure, vocabulary, complexity
  intellectualInterests: jsonb("intellectual_interests"), // Topics, subjects, domains
  cognitiveStyle: jsonb("cognitive_style"), // Analytical vs intuitive, detail vs big picture
  learningBehavior: jsonb("learning_behavior"), // How user improves over time
  documentPreferences: jsonb("document_preferences"), // Types and formats preferred

  collaborationStyle: jsonb("collaboration_style"), // Interaction with AI systems
  // Psychological indicators
  conceptualComplexity: text("conceptual_complexity"), // comfort with complex ideas
  attentionToDetail: integer("attention_to_detail"), // 1-10 scale
  creativityIndex: integer("creativity_index"), // 1-10 scale
  systematicThinking: integer("systematic_thinking"), // 1-10 scale
  // Behavioral metrics
  averageSessionLength: integer("average_session_length"), // in minutes
  totalDocumentsProcessed: integer("total_documents_processed"),
  preferredAIProvider: text("preferred_ai_provider"),
  productivityPattern: jsonb("productivity_pattern"), // Time of day, frequency patterns
  lastUpdated: timestamp("last_updated").defaultNow().notNull(),
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

export const insertCognitiveProfileSchema = createInsertSchema(cognitiveProfiles).pick({
  userId: true,
  writingPatterns: true,
  intellectualInterests: true,
  cognitiveStyle: true,
  learningBehavior: true,
  documentPreferences: true,

  collaborationStyle: true,
  conceptualComplexity: true,
  attentionToDetail: true,
  creativityIndex: true,
  systematicThinking: true,
  averageSessionLength: true,
  totalDocumentsProcessed: true,
  preferredAIProvider: true,
  productivityPattern: true,
});



// Types
export type InsertUser = z.infer<typeof insertUserSchema>;
export type User = typeof users.$inferSelect;
export type LoginData = Pick<InsertUser, "username" | "password">;

export type InsertDocument = z.infer<typeof insertDocumentSchema>;
export type Document = typeof documents.$inferSelect;

export type InsertAnalysis = z.infer<typeof insertAnalysisSchema>;
export type Analysis = typeof analyses.$inferSelect;

export type InsertComparison = z.infer<typeof insertComparisonSchema>;
export type Comparison = typeof comparisons.$inferSelect;

export type InsertUserActivity = z.infer<typeof insertUserActivitySchema>;
export type UserActivity = typeof userActivities.$inferSelect;

export type InsertCognitiveProfile = z.infer<typeof insertCognitiveProfileSchema>;
export type CognitiveProfile = typeof cognitiveProfiles.$inferSelect;



export type InsertCaseAssessment = z.infer<typeof insertCaseAssessmentSchema>;
export type CaseAssessment = typeof caseAssessments.$inferSelect;

// GPT Bypass Humanizer tables - using serial IDs to match existing schema
export const rewriteJobs = pgTable("rewrite_jobs", {
  id: serial("id").primaryKey(),
  inputText: text("input_text").notNull(),
  styleText: text("style_text"),
  contentMixText: text("content_mix_text"),
  customInstructions: text("custom_instructions"),
  selectedPresets: jsonb("selected_presets").$type<string[]>(),
  provider: text("provider").notNull(),
  chunks: jsonb("chunks").$type<TextChunk[]>(),
  selectedChunkIds: jsonb("selected_chunk_ids").$type<string[]>(),
  mixingMode: text("mixing_mode").$type<'style' | 'content' | 'both'>(),
  outputText: text("output_text"),
  inputAiScore: integer("input_ai_score"),
  outputAiScore: integer("output_ai_score"),
  status: text("status").notNull().default("pending"),
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

export const insertRewriteJobSchema = createInsertSchema(rewriteJobs).omit({
  id: true,
  createdAt: true,
});

export type InsertRewriteJob = z.infer<typeof insertRewriteJobSchema>;
export type RewriteJob = typeof rewriteJobs.$inferSelect;

export interface TextChunk {
  id: string;
  content: string;
  startWord: number;
  endWord: number;
  aiScore?: number;
}

export interface InstructionPreset {
  id: string;
  name: string;
  description: string;
  category: string;
  instruction: string;
}

export interface WritingSample {
  id: string;
  name: string;
  preview: string;
  content: string;
  category: string;
}

export interface AIProviderConfig {
  provider: 'openai' | 'anthropic' | 'deepseek' | 'perplexity';
  model?: string;
}

export interface RewriteRequest {
  inputText: string;
  styleText?: string;
  contentMixText?: string;
  customInstructions?: string;
  selectedPresets?: string[];
  provider: string;
  selectedChunkIds?: string[];
  mixingMode?: 'style' | 'content' | 'both';
}

export interface RewriteResponse {
  rewrittenText: string;
  inputAiScore: number;
  outputAiScore: number;
  jobId: string;
}

// Credit system tables for Stripe payment integration
export const userCredits = pgTable("user_credits", {
  id: serial("id").primaryKey(),
  userId: integer("user_id").references(() => users.id).notNull(),
  provider: text("provider").notNull(), // openai, anthropic, perplexity, deepseek
  credits: integer("credits").notNull().default(0), // word credits for this provider
  lastUpdated: timestamp("last_updated").defaultNow().notNull(),
});

export const creditTransactions = pgTable("credit_transactions", {
  id: serial("id").primaryKey(),
  userId: integer("user_id").references(() => users.id).notNull(),
  provider: text("provider").notNull(),
  amount: integer("amount").notNull(), // dollar amount in cents
  credits: integer("credits").notNull(), // word credits purchased/used
  transactionType: text("transaction_type").notNull(), // purchase, deduction
  stripeSessionId: text("stripe_session_id"),
  stripePaymentIntentId: text("stripe_payment_intent_id"),
  status: text("status").notNull().default("pending"), // pending, completed, failed
  metadata: jsonb("metadata"),
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

export const insertUserCreditsSchema = createInsertSchema(userCredits).omit({
  id: true,
  lastUpdated: true,
});

export const insertCreditTransactionSchema = createInsertSchema(creditTransactions).omit({
  id: true,
  createdAt: true,
});

export type InsertUserCredits = z.infer<typeof insertUserCreditsSchema>;
export type UserCredits = typeof userCredits.$inferSelect;

export type InsertCreditTransaction = z.infer<typeof insertCreditTransactionSchema>;
export type CreditTransaction = typeof creditTransactions.$inferSelect;

// Cross-chunk coherence system tables
export const coherenceDocuments = pgTable("coherence_documents", {
  id: serial("id").primaryKey(),
  documentId: text("document_id").notNull(),
  coherenceMode: text("coherence_mode").notNull(),
  globalState: jsonb("global_state").notNull(),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow().notNull(),
});

export const coherenceChunks = pgTable("coherence_chunks", {
  id: serial("id").primaryKey(),
  documentId: text("document_id").notNull(),
  coherenceMode: text("coherence_mode").notNull(),
  chunkIndex: integer("chunk_index").notNull(),
  chunkText: text("chunk_text"),
  evaluationResult: jsonb("evaluation_result"),
  stateAfter: jsonb("state_after"),
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

export const insertCoherenceDocumentSchema = createInsertSchema(coherenceDocuments).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export const insertCoherenceChunkSchema = createInsertSchema(coherenceChunks).omit({
  id: true,
  createdAt: true,
});

export type InsertCoherenceDocument = z.infer<typeof insertCoherenceDocumentSchema>;
export type CoherenceDocument = typeof coherenceDocuments.$inferSelect;
export type InsertCoherenceChunk = z.infer<typeof insertCoherenceChunkSchema>;
export type CoherenceChunk = typeof coherenceChunks.$inferSelect;

// Coherence mode types - using hyphenated format to match existing codebase
export type CoherenceModeType = 
  | "logical-consistency"
  | "logical-cohesiveness"
  | "scientific-explanatory"
  | "thematic-psychological"
  | "instructional"
  | "motivational"
  | "mathematical"
  | "philosophical";

// State templates by mode
export interface LogicalConsistencyState {
  mode: "logical-consistency";
  assertions: string[];
  negations: string[];
  disjoint_pairs: [string, string][];
}

export interface LogicalCohesivenessState {
  mode: "logical-cohesiveness";
  thesis: string;
  support_queue: string[];
  current_stage: "setup" | "development" | "conclusion";
  bridge_required: string;
}

export interface ScientificExplanatoryState {
  mode: "scientific-explanatory";
  causal_nodes: string[];
  causal_edges: { from: string; to: string; mechanism: string }[];
  level: string;
  active_feedback_loops: { name: string; nodes: string[] }[];
  mechanism_requirements: string[];
}

export interface ThematicPsychologicalState {
  mode: "thematic-psychological";
  dominant_affect: string;
  tempo: string;
  stance: string;
}

export interface InstructionalState {
  mode: "instructional";
  goal: string;
  steps_done: string[];
  prereqs: string[];
  open_loops: string[];
}

export interface MotivationalState {
  mode: "motivational";
  direction: "encourage" | "warn" | "pressure" | "reassure";
  intensity: number;
  target: string;
}

export interface MathematicalState {
  mode: "mathematical";
  givens: string[];
  proved: string[];
  goal: string;
  proof_method: string;
  dependencies: { step: string; depends_on: string[] }[];
}

export interface PhilosophicalState {
  mode: "philosophical";
  core_concepts: Record<string, string>;
  distinctions: string[];
  dialectic: { objections_raised: string[]; replies_pending: string[] };
  no_equivocation: string[];
}

export type CoherenceState =
  | LogicalConsistencyState
  | LogicalCohesivenessState
  | ScientificExplanatoryState
  | ThematicPsychologicalState
  | InstructionalState
  | MotivationalState
  | MathematicalState
  | PhilosophicalState;

export interface ChunkEvaluationResult {
  status: "preserved" | "weakened" | "broken";
  violations: { location: string; type: string; description: string }[];
  repairs: { location: string; suggestion: string }[];
  state_update: Partial<CoherenceState>;
}

// System instructions and prompts storage
export const systemInstructions = pgTable("system_instructions", {
  id: serial("id").primaryKey(),
  category: text("category").notNull(), // e.g., "development_guidelines", "coherence_prompts", "ui_patterns", "universal_design"
  title: text("title").notNull(),
  content: text("content").notNull(), // Full instruction/prompt text
  subcategory: text("subcategory"), // Optional: "condense_pathway", "synthesize_pathway", etc.
  version: text("version").default("1.0"),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow().notNull(),
});

export const insertSystemInstructionSchema = createInsertSchema(systemInstructions).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertSystemInstruction = z.infer<typeof insertSystemInstructionSchema>;
export type SystemInstruction = typeof systemInstructions.$inferSelect;

// Cross-Chunk Coherence (CC) Reconstruction Tables
// Store document-level state for multi-pass reconstruction
export const reconstructionDocuments = pgTable("reconstruction_documents", {
  id: serial("id").primaryKey(),
  userId: integer("user_id").references(() => users.id),
  title: text("title"),
  originalText: text("original_text").notNull(),
  wordCount: integer("word_count").notNull(),
  globalSkeleton: jsonb("global_skeleton"), // Stores skeleton extracted in Pass 1
  finalOutput: text("final_output"),
  finalWordCount: integer("final_word_count"), // Actual output word count
  validationResult: jsonb("validation_result"), // Stores Pass 3 validation results
  status: text("status").default("pending"), // pending, skeleton_extracted, chunks_processed, stitched, completed, failed
  
  // Length enforcement parameters
  targetMinWords: integer("target_min_words"), // Minimum target from user instructions
  targetMaxWords: integer("target_max_words"), // Maximum target from user instructions
  targetMidWords: integer("target_mid_words"), // Calculated midpoint
  lengthRatio: real("length_ratio"), // target_mid / input_words
  lengthMode: text("length_mode"), // heavy_compression, moderate_compression, maintain, moderate_expansion, heavy_expansion
  chunkTargetWords: integer("chunk_target_words"), // Per-chunk word target
  numChunks: integer("num_chunks"), // Total number of chunks
  currentChunk: integer("current_chunk").default(0), // Progress tracking
  
  audienceParameters: text("audience_parameters"),
  rigorLevel: text("rigor_level"),
  customInstructions: text("custom_instructions"),
  errorMessage: text("error_message"), // For failed jobs
  abortedAt: timestamp("aborted_at"), // When user aborted generation
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow().notNull(),
});

export const insertReconstructionDocumentSchema = createInsertSchema(reconstructionDocuments).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertReconstructionDocument = z.infer<typeof insertReconstructionDocumentSchema>;
export type ReconstructionDocument = typeof reconstructionDocuments.$inferSelect;

// Store per-chunk state for reconstruction
export const reconstructionChunks = pgTable("reconstruction_chunks", {
  id: serial("id").primaryKey(),
  documentId: integer("document_id").references(() => reconstructionDocuments.id).notNull(),
  chunkIndex: integer("chunk_index").notNull(),
  chunkInputText: text("chunk_input_text").notNull(),
  chunkInputWords: integer("chunk_input_words"), // Input word count
  chunkOutputText: text("chunk_output_text"),
  actualWords: integer("actual_words"), // Actual output word count
  
  // Per-chunk length targets
  targetWords: integer("target_words"),
  minWords: integer("min_words"),
  maxWords: integer("max_words"),
  
  chunkDelta: jsonb("chunk_delta"), // New claims, terms used, conflicts detected
  conflictsDetected: jsonb("conflicts_detected"), // Specific conflicts with skeleton
  status: text("status").default("pending"), // pending, processing, completed, conflict_flagged, retry, failed
  retryCount: integer("retry_count").default(0),
  errorMessage: text("error_message"),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow().notNull(),
});

export const insertReconstructionChunkSchema = createInsertSchema(reconstructionChunks).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertReconstructionChunk = z.infer<typeof insertReconstructionChunkSchema>;
export type ReconstructionChunk = typeof reconstructionChunks.$inferSelect;

// Store processing runs for audit/debugging
export const reconstructionRuns = pgTable("reconstruction_runs", {
  id: serial("id").primaryKey(),
  documentId: integer("document_id").references(() => reconstructionDocuments.id).notNull(),
  runType: text("run_type").notNull(), // 'skeleton', 'chunk_pass', 'stitch', 'repair'
  chunkIndex: integer("chunk_index"), // Only for chunk_pass runs
  runInput: jsonb("run_input"),
  runOutput: jsonb("run_output"),
  durationMs: integer("duration_ms"),
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

export const insertReconstructionRunSchema = createInsertSchema(reconstructionRuns).omit({
  id: true,
  createdAt: true,
});

export type InsertReconstructionRun = z.infer<typeof insertReconstructionRunSchema>;
export type ReconstructionRun = typeof reconstructionRuns.$inferSelect;

// Stitch results for Pass 3 global coherence validation
export const stitchResults = pgTable("stitch_results", {
  id: serial("id").primaryKey(),
  documentId: integer("document_id").references(() => reconstructionDocuments.id).notNull(),
  conflicts: jsonb("conflicts"), // Cross-chunk contradictions
  termDrift: jsonb("term_drift"), // Terminology inconsistencies
  missingPremises: jsonb("missing_premises"), // Claims without setup
  redundancies: jsonb("redundancies"), // Repeated points
  repairPlan: jsonb("repair_plan"), // Fixes to apply
  coherenceScore: text("coherence_score"), // 'pass' or 'needs_repair'
  finalValidation: jsonb("final_validation"), // Full validation result
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

export const insertStitchResultSchema = createInsertSchema(stitchResults).omit({
  id: true,
  createdAt: true,
});

export type InsertStitchResult = z.infer<typeof insertStitchResultSchema>;
export type StitchResultRecord = typeof stitchResults.$inferSelect;

// Chapter tracking for multi-chapter documents
export interface ChapterInfo {
  index: number;
  title: string;
  mainThesis: string;
  startWord: number;
  endWord: number;
  status: 'pending' | 'processed' | 'verified';
}

// Content addition request parsed from user instructions
export interface ContentAddition {
  type: 'concluding_chapter' | 'introduction' | 'summary' | 'custom';
  requirement: string;
  additional?: string;
}

// User instructions parsed from custom instructions
export interface UserInstructions {
  lengthTarget?: number;
  lengthConstraint?: 'no_less_than' | 'no_more_than' | 'approximately' | 'exactly';
  contentAdditions: ContentAddition[];
  mustAdd: string[];
  mustPreserve: string[];
  rawInstructions?: string;
}

// Global Skeleton type for Pass 1 extraction
export interface GlobalSkeleton {
  outline: string[];           // 8-20 numbered claims/sections
  thesis: string;              // Central argument
  keyTerms: { term: string; meaning: string }[];  // Terms with meanings
  commitmentLedger: { type: 'asserts' | 'rejects' | 'assumes'; claim: string }[];
  entities: { name: string; type: string; role: string }[];  // People, orgs, variables
  audienceParameters?: string;
  rigorLevel?: string;
  
  // User instructions capture (length, content additions, constraints)
  userInstructions?: UserInstructions;
  
  // Chapter tracking for multi-chapter documents
  chapters?: ChapterInfo[];
  chapterCount?: number;
}

// Chunk Delta type for Pass 2 tracking
export interface ChunkDelta {
  newClaimsIntroduced: string[];
  termsUsed: string[];
  conflictsDetected: { skeletonItem: string; chunkContent: string; description: string }[];
  ledgerAdditions: { type: 'asserts' | 'rejects' | 'assumes'; claim: string }[];
}

// Stitch Result type for Pass 3
export interface StitchResult {
  contradictions: { chunk1: number; chunk2: number; description: string }[];
  terminologyDrift: { term: string; chunk: number; originalMeaning: string; driftedMeaning: string }[];
  missingPremises: { location: number; description: string }[];
  redundancies: { chunks: number[]; description: string }[];
  repairPlan: { chunkIndex: number; repairAction: string }[];
}

// ═══════════════════════════════════════════════════════════════════════════
// HIERARCHICAL CROSS-CHUNK COHERENCE (HCC) ARCHITECTURE TABLES
// For processing book-length documents (100,000+ words)
// ═══════════════════════════════════════════════════════════════════════════

// HCC Document - top-level book/document
export const hccDocuments = pgTable("hcc_documents", {
  id: serial("id").primaryKey(),
  userId: integer("user_id").references(() => users.id),
  title: text("title"),
  originalText: text("original_text").notNull(),
  wordCount: integer("word_count").notNull(),
  structureMap: jsonb("structure_map"), // detected/imposed hierarchy
  bookSkeleton: jsonb("book_skeleton"), // ~2000 tokens max
  finalOutput: text("final_output"),
  targetMinWords: integer("target_min_words"),
  targetMaxWords: integer("target_max_words"),
  lengthRatio: text("length_ratio"), // numeric stored as text for precision
  lengthMode: text("length_mode"), // heavy_compression, moderate_compression, maintain, moderate_expansion, heavy_expansion
  status: text("status").default("pending"), // pending, structure_detected, skeletons_extracted, processing, stitching, complete, failed
  customInstructions: text("custom_instructions"),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow().notNull(),
});

export const insertHccDocumentSchema = createInsertSchema(hccDocuments).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertHccDocument = z.infer<typeof insertHccDocumentSchema>;
export type HccDocument = typeof hccDocuments.$inferSelect;

// HCC Parts (or virtual parts ~25,000 words each)
export const hccParts = pgTable("hcc_parts", {
  id: serial("id").primaryKey(),
  documentId: integer("document_id").references(() => hccDocuments.id).notNull(),
  partIndex: integer("part_index").notNull(),
  partTitle: text("part_title"),
  originalText: text("original_text").notNull(),
  wordCount: integer("word_count").notNull(),
  partSkeleton: jsonb("part_skeleton"), // ~1000 tokens
  compressedBookSkeleton: jsonb("compressed_book_skeleton"), // inherited, ~500 tokens
  partOutput: text("part_output"),
  partDelta: jsonb("part_delta"), // net contribution to argument
  status: text("status").default("pending"),
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

export const insertHccPartSchema = createInsertSchema(hccParts).omit({
  id: true,
  createdAt: true,
});

export type InsertHccPart = z.infer<typeof insertHccPartSchema>;
export type HccPart = typeof hccParts.$inferSelect;

// HCC Chapters (or virtual chapters ~5,000 words each)
export const hccChapters = pgTable("hcc_chapters", {
  id: serial("id").primaryKey(),
  partId: integer("part_id").references(() => hccParts.id).notNull(),
  documentId: integer("document_id").references(() => hccDocuments.id).notNull(),
  chapterIndex: integer("chapter_index").notNull(),
  chapterTitle: text("chapter_title"),
  originalText: text("original_text").notNull(),
  wordCount: integer("word_count").notNull(),
  chapterSkeleton: jsonb("chapter_skeleton"), // ~700 tokens
  compressedPartSkeleton: jsonb("compressed_part_skeleton"), // inherited, ~300 tokens
  chapterOutput: text("chapter_output"),
  chapterDelta: jsonb("chapter_delta"), // net contribution to part
  status: text("status").default("pending"),
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

export const insertHccChapterSchema = createInsertSchema(hccChapters).omit({
  id: true,
  createdAt: true,
});

export type InsertHccChapter = z.infer<typeof insertHccChapterSchema>;
export type HccChapter = typeof hccChapters.$inferSelect;

// HCC Chunks (with length enforcement)
export const hccChunks = pgTable("hcc_chunks", {
  id: serial("id").primaryKey(),
  chapterId: integer("chapter_id").references(() => hccChapters.id).notNull(),
  documentId: integer("document_id").references(() => hccDocuments.id).notNull(),
  chunkIndex: integer("chunk_index").notNull(),
  chunkInputText: text("chunk_input_text").notNull(),
  chunkInputWords: integer("chunk_input_words").notNull(),
  chunkOutputText: text("chunk_output_text"),
  chunkOutputWords: integer("chunk_output_words"),
  targetWords: integer("target_words"), // calculated per-chunk target
  minWords: integer("min_words"), // target * 0.85
  maxWords: integer("max_words"), // target * 1.15
  chunkDelta: jsonb("chunk_delta"),
  conflictsDetected: jsonb("conflicts_detected"),
  status: text("status").default("pending"),
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

export const insertHccChunkSchema = createInsertSchema(hccChunks).omit({
  id: true,
  createdAt: true,
});

export type InsertHccChunk = z.infer<typeof insertHccChunkSchema>;
export type HccChunk = typeof hccChunks.$inferSelect;

// HCC Processing Types
export interface HccBookSkeleton {
  masterThesis: string;
  majorDivisions: { title: string; summary: string }[];
  globalTerms: { term: string; definition: string }[];
  coreCommitments: { type: 'asserts' | 'rejects' | 'assumes'; claim: string }[];
  crossReferences: { from: string; to: string; relationship: string }[];
}

export interface HccPartSkeleton {
  partThesis: string;
  chapterSummaries: { title: string; summary: string }[];
  partSpecificTerms: { term: string; definition: string }[];
  inheritedCommitments: { type: 'asserts' | 'rejects' | 'assumes'; claim: string }[];
}

export interface HccChapterSkeleton {
  chapterThesis: string;
  sectionOutline: string[];
  chapterTerms: { term: string; definition: string }[];
  inheritedContext: string; // compressed part skeleton
}

export interface HccDelta {
  netContribution: string;
  newCommitments: { type: 'asserts' | 'rejects' | 'assumes'; claim: string }[];
  conflictsResolved: string[];
  conflictsFlagged: string[];
  crossReferences: { to: string; type: string }[];
}

export interface LengthEnforcementConfig {
  targetMinWords: number;
  targetMaxWords: number;
  targetMidWords: number;
  lengthRatio: number;
  lengthMode: 'heavy_compression' | 'moderate_compression' | 'maintain' | 'moderate_expansion' | 'heavy_expansion';
}

// ═══════════════════════════════════════════════════════════════════════════
// FULL PIPELINE CROSS-CHUNK COHERENCE (FPCC) ARCHITECTURE TABLES
// 4-Stage Pipeline: Reconstruction → Objections → Responses → Bullet-proof
// With Vertical Coherence (VC) per stage and Horizontal Coherence (HC) across stages
// ═══════════════════════════════════════════════════════════════════════════

// Pipeline job tracking - orchestrates the 4-stage pipeline
export const pipelineJobs = pgTable("pipeline_jobs", {
  id: serial("id").primaryKey(),
  userId: integer("user_id").references(() => users.id),
  
  // Original input
  originalText: text("original_text").notNull(),
  originalWordCount: integer("original_word_count").notNull(),
  
  // User parameters
  customInstructions: text("custom_instructions"),
  targetAudience: text("target_audience"),
  objective: text("objective"),
  
  // Stage outputs
  reconstructionOutput: text("reconstruction_output"),
  objectionsOutput: text("objections_output"),
  responsesOutput: text("responses_output"),
  bulletproofOutput: text("bulletproof_output"),
  
  // Skeletons (JSONB for flexibility)
  skeleton1: jsonb("skeleton_1"), // Reconstruction skeleton
  skeleton2: jsonb("skeleton_2"), // Objections skeleton
  skeleton3: jsonb("skeleton_3"), // Responses skeleton
  skeleton4: jsonb("skeleton_4"), // Bullet-proof skeleton
  
  // Progress tracking
  currentStage: integer("current_stage").default(1), // 1-4
  stageStatus: text("stage_status").default("pending"), // pending, skeleton_extraction, chunk_processing, stitching, complete
  totalStages: integer("total_stages").default(4),
  
  // Stage word counts
  reconstructionWords: integer("reconstruction_words"),
  objectionsWords: integer("objections_words"),
  responsesWords: integer("responses_words"),
  bulletproofWords: integer("bulletproof_words"),
  
  // HC check results
  hcCheckResults: jsonb("hc_check_results"),
  hcViolations: jsonb("hc_violations"), // Array of violation objects
  hcRepairAttempts: integer("hc_repair_attempts").default(0),
  
  // Timing
  stage1StartTime: timestamp("stage1_start_time"),
  stage1EndTime: timestamp("stage1_end_time"),
  stage2StartTime: timestamp("stage2_start_time"),
  stage2EndTime: timestamp("stage2_end_time"),
  stage3StartTime: timestamp("stage3_start_time"),
  stage3EndTime: timestamp("stage3_end_time"),
  stage4StartTime: timestamp("stage4_start_time"),
  stage4EndTime: timestamp("stage4_end_time"),
  hcCheckTime: timestamp("hc_check_time"),
  
  // Final status
  status: text("status").default("pending"), // pending, running, paused, complete, completed_with_warnings, failed
  errorMessage: text("error_message"),
  
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow().notNull(),
});

export const insertPipelineJobSchema = createInsertSchema(pipelineJobs).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertPipelineJob = z.infer<typeof insertPipelineJobSchema>;
export type PipelineJob = typeof pipelineJobs.$inferSelect;

// Stage-specific chunk tracking
export const pipelineChunks = pgTable("pipeline_chunks", {
  id: serial("id").primaryKey(),
  jobId: integer("job_id").references(() => pipelineJobs.id).notNull(),
  stage: integer("stage").notNull(), // 1-4
  chunkIndex: integer("chunk_index").notNull(),
  
  chunkInputText: text("chunk_input_text"),
  chunkOutputText: text("chunk_output_text"),
  chunkDelta: jsonb("chunk_delta"), // Stage-specific delta information
  
  targetWords: integer("target_words"),
  actualWords: integer("actual_words"),
  minWords: integer("min_words"),
  maxWords: integer("max_words"),
  
  status: text("status").default("pending"), // pending, processing, completed, retrying, failed
  retryCount: integer("retry_count").default(0),
  errorMessage: text("error_message"),
  
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

export const insertPipelineChunkSchema = createInsertSchema(pipelineChunks).omit({
  id: true,
  createdAt: true,
});

export type InsertPipelineChunk = z.infer<typeof insertPipelineChunkSchema>;
export type PipelineChunk = typeof pipelineChunks.$inferSelect;

// Objection tracking (for Stage 2-4 coherence)
export const pipelineObjections = pgTable("pipeline_objections", {
  id: serial("id").primaryKey(),
  jobId: integer("job_id").references(() => pipelineJobs.id).notNull(),
  objectionIndex: integer("objection_index").notNull(), // 1-25
  
  // Stage 2: Objection details
  claimTargeted: text("claim_targeted"), // Exact quote or paraphrase from reconstruction
  claimLocation: text("claim_location"), // Section/paragraph reference
  objectionType: text("objection_type"), // logical, empirical, conceptual, methodological, practical
  objectionText: text("objection_text"), // The objection itself
  severity: text("severity"), // fatal, serious, moderate, minor
  
  // Stage 2: Initial response
  initialResponse: text("initial_response"),
  
  // Stage 3: Enhanced response
  enhancedResponse: text("enhanced_response"),
  enhancementNotes: text("enhancement_notes"), // What was improved
  
  // Stage 4: Integration tracking
  integratedInSection: text("integrated_in_section"), // Where it appears in bullet-proof
  integrationStrategy: text("integration_strategy"), // preemptive, inline, footnote, structural
  integrationVerified: boolean("integration_verified").default(false),
  
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

export const insertPipelineObjectionSchema = createInsertSchema(pipelineObjections).omit({
  id: true,
  createdAt: true,
});

export type InsertPipelineObjection = z.infer<typeof insertPipelineObjectionSchema>;
export type PipelineObjection = typeof pipelineObjections.$inferSelect;

// ═══════════════════════════════════════════════════════════════════════════
// PIPELINE SKELETON TYPES
// ═══════════════════════════════════════════════════════════════════════════

// Stage 1: Reconstruction skeleton (extends GlobalSkeleton)
export interface PipelineSkeleton1 extends GlobalSkeleton {
  // Inherits: outline, thesis, keyTerms, commitmentLedger, entities
  // Additional for pipeline tracking:
  documentWordCount: number;
  reconstructionWordCount: number;
}

// Stage 2: Objections skeleton
export interface PipelineSkeleton2 {
  claimsToTarget: { claimIndex: number; claim: string; location: string }[];
  claimLocations: { [claimIndex: number]: string };
  objectionTypes: {
    logical: number[];
    empirical: number[];
    conceptual: number[];
    methodological: number[];
    practical: number[];
  };
  severityDistribution: {
    fatal: number[];
    serious: number[];
    moderate: number[];
    minor: number[];
  };
  inheritedCommitments: { type: 'asserts' | 'rejects' | 'assumes'; claim: string }[];
  objectionSummaries: { index: number; summary: string }[];
  responseSummaries: { index: number; summary: string }[];
}

// Stage 3: Responses skeleton
export interface PipelineSkeleton3 {
  objectionsToAddress: { index: number; summary: string }[];
  initialResponses: { index: number; summary: string }[];
  responseGaps: { index: number; gap: string }[];
  enhancementStrategy: {
    index: number;
    strategy: 'additional_evidence' | 'deeper_analysis' | 'practical_examples' | 'concession_rebuttal';
    notes: string;
  }[];
  enhancedResponseSummaries: { index: number; summary: string }[];
  newCommitments: { type: 'asserts' | 'rejects' | 'assumes'; claim: string }[];
  concessionsMade: { objectionIndex: number; concession: string }[];
  inheritedSkeleton1: Partial<PipelineSkeleton1>;
  inheritedSkeleton2: Partial<PipelineSkeleton2>;
}

// Stage 4: Bullet-proof skeleton
export interface PipelineSkeleton4 {
  originalStructure: { sectionIndex: number; sectionTitle: string; wordCount: number }[];
  integrationMap: { sectionIndex: number; responseIndices: number[] }[];
  integrationStrategy: {
    responseIndex: number;
    strategy: 'preemptive' | 'inline' | 'footnote' | 'structural';
    targetSection: number;
  }[];
  concessionsToIncorporate: { objectionIndex: number; concession: string }[];
  strengtheningAdditions: { responseIndex: number; addition: string }[];
  commitmentReconciliation: {
    originalCommitment: string;
    status: 'preserved' | 'revised' | 'defended';
    notes: string;
  }[];
  lengthTarget: { min: number; max: number; target: number };
  keyTerms: { term: string; definition: string }[];
  inheritedSkeletons: {
    skeleton1: Partial<PipelineSkeleton1>;
    skeleton2: Partial<PipelineSkeleton2>;
    skeleton3: Partial<PipelineSkeleton3>;
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// HORIZONTAL COHERENCE (HC) CHECK TYPES
// ═══════════════════════════════════════════════════════════════════════════

export interface HCViolation {
  type: 'commitment_missing' | 'objection_not_addressed' | 'response_not_integrated' | 'terminology_drift' | 'contradiction';
  severity: 'error' | 'warning';
  description: string;
  details: {
    commitment?: string;
    objectionIndex?: number;
    responseIndex?: number;
    term?: string;
    originalDefinition?: string;
    newDefinition?: string;
    location?: string;
  };
}

export interface HCCheckResult {
  passed: boolean;
  violations: HCViolation[];
  summary: {
    total: number;
    errors: number;
    warnings: number;
    commitmentsMissing: number;
    objectionsNotAddressed: number;
    responsesNotIntegrated: number;
    terminologyDrifts: number;
  };
  repairPlan?: {
    sectionsToRevise: number[];
    violationsToFix: number[];
    instructions: string;
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// RECURSIVE OBJECTION-PROOF REWRITE TYPES
// For iterative rewrites with cross-chunk coherence
// ═══════════════════════════════════════════════════════════════════════════

export interface ObjectionProofIteration {
  id: string;
  parentId: string | null;
  version: number;
  inputText: string;
  outputText: string;
  wordCount: number;
  targetWordCount: number | null;
  customInstructions: string | null;
  globalSkeleton: GlobalSkeleton | null;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  createdAt: Date;
}

export interface RecursiveRewriteRequest {
  text: string;
  targetWordCount?: number;
  customInstructions?: string;
  parentIterationId?: string;
}

export interface RecursiveRewriteResult {
  success: boolean;
  iteration: ObjectionProofIteration;
  processingStats?: {
    inputWords: number;
    outputWords: number;
    chunksProcessed: number;
    coherenceScore?: string;
  };
  error?: string;
}
