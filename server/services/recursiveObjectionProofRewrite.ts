import Anthropic from "@anthropic-ai/sdk";
import { v4 as uuidv4 } from "uuid";
import {
  ObjectionProofIteration,
  RecursiveRewriteRequest,
  RecursiveRewriteResult,
  GlobalSkeleton,
  ChunkDelta,
} from "@shared/schema";
import {
  extractGlobalSkeleton,
  smartChunk,
  reconstructChunkConstrained,
  stitchAndValidate,
  parseTargetLength,
  calculateLengthConfig,
} from "./crossChunkCoherence";

type LengthMode = 'heavy_compression' | 'moderate_compression' | 'maintain' | 'moderate_expansion' | 'heavy_expansion';

interface LengthConfig {
  targetMin: number;
  targetMax: number;
  targetMid: number;
  lengthRatio: number;
  lengthMode: LengthMode;
  chunkTargetWords: number;
}

let _anthropic: Anthropic | null = null;
function getAnthropic(): Anthropic {
  if (!_anthropic) _anthropic = new Anthropic();
  return _anthropic;
}

const PRIMARY_MODEL = "claude-sonnet-4-5-20250929";
const SHORT_TEXT_THRESHOLD = 1500;
const TARGET_CHUNK_SIZE = 800;

interface IterationStore {
  [key: string]: ObjectionProofIteration;
}

const iterationStore: IterationStore = {};

function countWords(text: string): number {
  return text.split(/\s+/).filter((w) => w.length > 0).length;
}

export async function performRecursiveRewrite(
  request: RecursiveRewriteRequest,
  onProgress?: (phase: string, message: string, progress?: number) => void
): Promise<RecursiveRewriteResult> {
  const iterationId = uuidv4();
  const inputWords = countWords(request.text);

  const iteration: ObjectionProofIteration = {
    id: iterationId,
    parentId: request.parentIterationId || null,
    version: request.parentIterationId
      ? (iterationStore[request.parentIterationId]?.version || 0) + 1
      : 1,
    inputText: request.text,
    outputText: "",
    wordCount: inputWords,
    targetWordCount: request.targetWordCount || null,
    customInstructions: request.customInstructions || null,
    globalSkeleton: null,
    status: "processing",
    createdAt: new Date(),
  };

  iterationStore[iterationId] = iteration;

  try {
    onProgress?.("initializing", "Starting recursive objection-proof rewrite...", 0);

    const parsedTarget = parseTargetLength(request.customInstructions || null);
    const targetMin = request.targetWordCount
      ? Math.round(request.targetWordCount * 0.9)
      : parsedTarget?.targetMin || null;
    const targetMax = request.targetWordCount
      ? Math.round(request.targetWordCount * 1.1)
      : parsedTarget?.targetMax || null;

    const lengthConfig = calculateLengthConfig(
      inputWords,
      targetMin,
      targetMax,
      request.customInstructions || null
    );

    console.log(`[RECURSIVE-OP] Processing ${inputWords} words, target: ${lengthConfig.targetMid} words, mode: ${lengthConfig.lengthMode}`);

    if (inputWords < SHORT_TEXT_THRESHOLD) {
      onProgress?.("processing", "Processing short text directly...", 30);
      const result = await processShortText(
        request.text,
        lengthConfig.targetMid,
        request.customInstructions || null
      );

      iteration.outputText = result.output;
      iteration.wordCount = countWords(result.output);
      iteration.status = "completed";
      iterationStore[iterationId] = iteration;

      return {
        success: true,
        iteration,
        processingStats: {
          inputWords,
          outputWords: iteration.wordCount,
          chunksProcessed: 1,
          coherenceScore: "short-text",
        },
      };
    }

    onProgress?.("skeleton", "Extracting global skeleton for coherence...", 10);
    const skeleton = await extractGlobalSkeleton(
      request.text,
      undefined,
      undefined,
      request.customInstructions || undefined
    );
    iteration.globalSkeleton = skeleton;

    onProgress?.("chunking", "Chunking text for processing...", 20);
    const chunks = smartChunk(request.text);
    const numChunks = chunks.length;

    console.log(`[RECURSIVE-OP] Split into ${numChunks} chunks`);

    const processedChunks: Array<{ text: string; delta: ChunkDelta }> = [];
    let runningWordCount = 0;

    for (let i = 0; i < chunks.length; i++) {
      const chunk = chunks[i];
      const chunkTargetWords = Math.ceil(lengthConfig.targetMid / numChunks);
      const progress = 20 + Math.floor(((i + 1) / numChunks) * 60);

      onProgress?.(
        "processing",
        `Processing chunk ${i + 1} of ${numChunks}...`,
        progress
      );

      const result = await reconstructChunkConstrained(
        chunk.text,
        i,
        numChunks,
        skeleton,
        undefined,
        chunkTargetWords,
        undefined,
        lengthConfig
      );

      processedChunks.push({
        text: result.outputText,
        delta: result.delta,
      });

      runningWordCount += countWords(result.outputText);

      if (i < chunks.length - 1) {
        await new Promise((resolve) => setTimeout(resolve, 1500));
      }
    }

    onProgress?.("stitching", "Running global consistency check...", 85);
    const { finalOutput, stitchResult } = await stitchAndValidate(
      skeleton,
      processedChunks
    );

    onProgress?.("finalizing", "Finalizing output...", 95);

    iteration.outputText = finalOutput;
    iteration.wordCount = countWords(finalOutput);
    iteration.status = "completed";
    iterationStore[iterationId] = iteration;

    return {
      success: true,
      iteration,
      processingStats: {
        inputWords,
        outputWords: iteration.wordCount,
        chunksProcessed: numChunks,
        coherenceScore: stitchResult.contradictions.length === 0 ? "pass" : "needs_review",
      },
    };
  } catch (error: any) {
    console.error("[RECURSIVE-OP] Error:", error);
    iteration.status = "failed";
    iteration.outputText = "";
    iterationStore[iterationId] = iteration;

    return {
      success: false,
      iteration,
      error: error.message || "Processing failed",
    };
  }
}

async function processShortText(
  text: string,
  targetWords: number,
  customInstructions: string | null
): Promise<{ output: string }> {
  const anthropic = getAnthropic();

  const prompt = `You are rewriting text to make it stronger against objections while maintaining coherence.

INPUT TEXT:
${text}

TARGET WORD COUNT: approximately ${targetWords} words

${customInstructions ? `CUSTOM INSTRUCTIONS:\n${customInstructions}\n` : ""}

REWRITE REQUIREMENTS:
1. Maintain the original argument's structure and flow
2. Strengthen weak points that could be challenged
3. Add necessary qualifications and evidence
4. Preserve key terminology and concepts
5. Ensure logical coherence throughout
6. Hit the target word count (within 10%)

OUTPUT: Provide ONLY the rewritten text, no meta-commentary or explanations.`;

  const response = await anthropic.messages.create({
    model: PRIMARY_MODEL,
    max_tokens: 8000,
    temperature: 0.3,
    messages: [{ role: "user", content: prompt }],
  });

  const output =
    response.content[0].type === "text" ? response.content[0].text : "";

  return { output: output.trim() };
}


export function getIteration(id: string): ObjectionProofIteration | null {
  return iterationStore[id] || null;
}

export function getIterationHistory(
  currentId: string
): ObjectionProofIteration[] {
  const history: ObjectionProofIteration[] = [];
  let current = iterationStore[currentId];

  while (current) {
    history.unshift(current);
    if (current.parentId) {
      current = iterationStore[current.parentId];
    } else {
      break;
    }
  }

  return history;
}

export function clearIterationStore(): void {
  Object.keys(iterationStore).forEach((key) => delete iterationStore[key]);
}
