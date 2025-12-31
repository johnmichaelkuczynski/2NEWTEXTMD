import Anthropic from '@anthropic-ai/sdk';

interface Position {
  author: string;
  claim: string;
  category: string;
  originalLine: string;
  lineNumber: number;
}

interface RankedPosition extends Position {
  significanceScore: number;
  significanceReason: string;
}

interface PositionDefense {
  position: RankedPosition;
  defense: string;
}

interface PositionListResult {
  success: boolean;
  output: string;
  positionsProcessed: number;
  positionsSelected: number;
  totalPositions: number;
  error?: string;
}

interface DefenseFormat {
  sentenceCount: number | null;
  wordCount: { min: number; max: number } | null;
  pageLength: boolean;
  paragraphCount: number | null;
  startWithQuotedClaim: boolean;
}

export function isPositionList(text: string): boolean {
  const lines = text.trim().split('\n').filter(l => l.trim().length > 0);
  // Accept short lists (as few as 1 line) if they match pipe-delimited format
  if (lines.length < 1) return false;

  let pipeDelimitedCount = 0;
  const linesToCheck = lines.slice(0, Math.min(20, lines.length));
  for (const line of linesToCheck) {
    const parts = line.split('|').map(p => p.trim());
    // A valid position line has at least 3 parts with meaningful content
    if (parts.length >= 3 && parts[1].length > 10) {
      pipeDelimitedCount++;
    }
  }

  // For very short inputs (1-9 lines), require ALL lines to be pipe-delimited
  // For longer inputs (10+), require 70% to be pipe-delimited
  if (lines.length < 10) {
    return pipeDelimitedCount === linesToCheck.length && pipeDelimitedCount > 0;
  }
  return pipeDelimitedCount >= linesToCheck.length * 0.7;
}

function parsePositions(text: string): Position[] {
  const lines = text.trim().split('\n').filter(l => l.trim().length > 0);
  const positions: Position[] = [];

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    const parts = line.split('|').map(p => p.trim());

    if (parts.length >= 3) {
      positions.push({
        author: parts[0],
        claim: parts[1],
        category: parts[2],
        originalLine: line,
        lineNumber: i + 1
      });
    } else if (parts.length === 2 && parts[1].length > 10) {
      positions.push({
        author: parts[0],
        claim: parts[1],
        category: 'Uncategorized',
        originalLine: line,
        lineNumber: i + 1
      });
    }
  }

  return positions;
}

/**
 * FIX #1: Properly parse selection count, recognizing "EACH", "ALL", "EVERY"
 */
function parseSelectionCount(customInstructions: string, totalPositions: number): number {
  const upper = customInstructions.toUpperCase();

  // Check for "ALL" or "EACH" or "EVERY" indicators - process everything
  const allIndicators = [
    /\bEACH\s+(OF\s+)?(THE\s+)?(FOLLOWING\s+)?CLAIM/i,
    /\bALL\s+(OF\s+)?(THE\s+)?(FOLLOWING\s+)?CLAIM/i,
    /\bEVERY\s+CLAIM/i,
    /\bEACH\s+POSITION/i,
    /\bALL\s+POSITIONS/i,
    /\bDEFEND\s+EACH/i,
    /\bDEFEND\s+ALL/i,
  ];

  for (const pattern of allIndicators) {
    if (pattern.test(customInstructions)) {
      console.log(`[POSITION-LIST] Detected "process all" instruction, returning ${totalPositions}`);
      return totalPositions;
    }
  }

  // Check for explicit numeric selection
  const numericPatterns = [
    /(?:top|best|most\s+(?:significant|important|crucial))\s+(\d+)/i,
    /(\d+)\s+(?:most|best|top)/i,
    /select\s+(\d+)/i,
    /defend\s+(?:the\s+)?(\d+)\s+(?:most|best|top)/i,
    /focus\s+on\s+(\d+)/i,
  ];

  for (const pattern of numericPatterns) {
    const match = customInstructions.match(pattern);
    if (match) {
      return parseInt(match[1]);
    }
  }

  // Default: process ALL positions (changed from 25)
  console.log(`[POSITION-LIST] No selection count found, defaulting to ALL (${totalPositions})`);
  return totalPositions;
}

/**
 * FIX #2: Parse defense format requirements from custom instructions
 */
function parseDefenseFormat(customInstructions: string): DefenseFormat {
  const format: DefenseFormat = {
    sentenceCount: null,
    wordCount: null,
    pageLength: false,
    paragraphCount: null,
    startWithQuotedClaim: false
  };

  // Check for quoted claim requirement
  if (/START\s+WITH\s+(THE\s+)?CLAIM.*QUOTATION|QUOTE.*CLAIM|CLAIM.*QUOTE|".*CLAIM.*"/i.test(customInstructions)) {
    format.startWithQuotedClaim = true;
    console.log(`[POSITION-LIST] Detected requirement: start with quoted claim`);
  }

  // Check for sentence count
  const sentencePatterns = [
    /(\d+)\s*[-–]?\s*SENTENCE/i,
    /(\w+)\s+SENTENCE\s+DEFENSE/i,  // "FIVE SENTENCE DEFENSE"
  ];

  const wordToNum: Record<string, number> = {
    'ONE': 1, 'TWO': 2, 'THREE': 3, 'FOUR': 4, 'FIVE': 5,
    'SIX': 6, 'SEVEN': 7, 'EIGHT': 8, 'NINE': 9, 'TEN': 10
  };

  for (const pattern of sentencePatterns) {
    const match = customInstructions.match(pattern);
    if (match) {
      const numStr = match[1].toUpperCase();
      format.sentenceCount = wordToNum[numStr] || parseInt(numStr) || null;
      if (format.sentenceCount) {
        console.log(`[POSITION-LIST] Detected sentence count requirement: ${format.sentenceCount}`);
      }
      break;
    }
  }

  // Check for page length
  if (/ONE\s+PAGE|1\s+PAGE|FULL\s+PAGE|PAGE\s+PER/i.test(customInstructions)) {
    format.pageLength = true;
    console.log(`[POSITION-LIST] Detected page-length requirement`);
  }

  // Check for paragraph count
  const paraMatch = customInstructions.match(/(\d+)\s*PARAGRAPH/i) ||
                    customInstructions.match(/(\w+)\s+PARAGRAPH/i);
  if (paraMatch) {
    const numStr = paraMatch[1].toUpperCase();
    format.paragraphCount = wordToNum[numStr] || parseInt(numStr) || null;
    if (format.paragraphCount) {
      console.log(`[POSITION-LIST] Detected paragraph count requirement: ${format.paragraphCount}`);
    }
  }

  // Check for word count range
  const wordMatch = customInstructions.match(/(\d+)\s*[-–to]+\s*(\d+)\s*WORD/i);
  if (wordMatch) {
    format.wordCount = { min: parseInt(wordMatch[1]), max: parseInt(wordMatch[2]) };
    console.log(`[POSITION-LIST] Detected word count requirement: ${format.wordCount.min}-${format.wordCount.max}`);
  }

  return format;
}

/**
 * FIX #3: Build format instructions for defense generation prompt
 */
function buildFormatInstructions(format: DefenseFormat, customInstructions: string): string {
  const parts: string[] = [];

  // Start with quoted claim requirement (CRITICAL)
  if (format.startWithQuotedClaim) {
    parts.push('START each defense by reproducing the EXACT claim text in quotation marks on its own line.');
  }

  // Length requirements
  if (format.sentenceCount) {
    parts.push(`Each defense must be EXACTLY ${format.sentenceCount} sentences. Count carefully.`);
  } else if (format.pageLength) {
    parts.push('Each defense should be approximately one page (400-500 words).');
  } else if (format.wordCount) {
    parts.push(`Each defense should be ${format.wordCount.min}-${format.wordCount.max} words.`);
  } else if (format.paragraphCount) {
    parts.push(`Each defense should be exactly ${format.paragraphCount} paragraph(s).`);
  } else {
    // Default
    parts.push('Each defense should be 150-300 words.');
  }

  // Include original custom instructions if they contain other guidance
  if (customInstructions && !format.sentenceCount && !format.pageLength) {
    parts.push(`\nADDITIONAL USER GUIDANCE: ${customInstructions}`);
  }

  return parts.join('\n');
}

async function rankPositions(
  positions: Position[],
  customInstructions: string,
  targetCount: number
): Promise<RankedPosition[]> {
  // FIX #4: Skip ranking entirely if processing all positions
  if (targetCount >= positions.length) {
    console.log(`[POSITION-LIST] Processing all ${positions.length} positions, skipping ranking`);
    return positions.map((p, i) => ({
      ...p,
      significanceScore: 100 - i, // Preserve original order with descending scores
      significanceReason: 'Processing all positions as requested'
    }));
  }

  const anthropic = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });

  const batchSize = 50;
  const batches: Position[][] = [];
  for (let i = 0; i < positions.length; i += batchSize) {
    batches.push(positions.slice(i, i + batchSize));
  }

  const allRanked: RankedPosition[] = [];

  for (let batchIdx = 0; batchIdx < batches.length; batchIdx++) {
    const batch = batches[batchIdx];
    console.log(`[POSITION-LIST] Ranking batch ${batchIdx + 1}/${batches.length} (${batch.length} positions)`);

    const positionList = batch.map((p, i) => 
      `${i + 1}. [${p.category}] ${p.claim}`
    ).join('\n');

    const systemPrompt = `You are an expert at evaluating philosophical and theoretical positions for their significance, novelty, and intellectual importance.

Score each position from 1-100 based on:
- Intellectual significance and depth
- Novelty or counter-intuitiveness
- Importance to the overall argument framework
- Whether it challenges established thinking

${customInstructions ? `USER'S CRITERIA FOR SELECTION: ${customInstructions}` : ''}

Return a JSON array with scores for each position:
[
  {"index": 1, "score": 85, "reason": "Brief reason"},
  {"index": 2, "score": 72, "reason": "Brief reason"},
  ...
]

Be discriminating - use the full range from 20-95. Reserve 90+ for truly exceptional positions.`;

    const response = await anthropic.messages.create({
      model: "claude-sonnet-4-20250514",
      max_tokens: 4000,
      system: systemPrompt,
      messages: [{
        role: "user",
        content: `Score these ${batch.length} positions:\n\n${positionList}`
      }]
    });

    const responseText = (response.content[0] as any).text;

    try {
      const jsonMatch = responseText.match(/\[[\s\S]*\]/);
      if (jsonMatch) {
        const scores = JSON.parse(jsonMatch[0]);
        for (const scoreObj of scores) {
          const idx = scoreObj.index - 1;
          if (idx >= 0 && idx < batch.length) {
            allRanked.push({
              ...batch[idx],
              significanceScore: scoreObj.score || 50,
              significanceReason: scoreObj.reason || ''
            });
          }
        }
      }
    } catch (parseError) {
      console.error('[POSITION-LIST] Failed to parse ranking response, using defaults');
      for (const pos of batch) {
        allRanked.push({
          ...pos,
          significanceScore: 50,
          significanceReason: 'Default score'
        });
      }
    }

    if (batchIdx < batches.length - 1) {
      await new Promise(resolve => setTimeout(resolve, 500));
    }
  }

  allRanked.sort((a, b) => b.significanceScore - a.significanceScore);

  return allRanked.slice(0, targetCount);
}

/**
 * FIX #5: Completely rewritten defense generation with proper format enforcement
 */
async function generateDefenses(
  positions: RankedPosition[],
  customInstructions: string,
  format: DefenseFormat
): Promise<PositionDefense[]> {
  const anthropic = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });

  const batchSize = 5;
  const batches: RankedPosition[][] = [];
  for (let i = 0; i < positions.length; i += batchSize) {
    batches.push(positions.slice(i, i + batchSize));
  }

  const defenses: PositionDefense[] = [];
  const formatInstructions = buildFormatInstructions(format, customInstructions);

  for (let batchIdx = 0; batchIdx < batches.length; batchIdx++) {
    const batch = batches[batchIdx];
    console.log(`[POSITION-LIST] Generating defenses for batch ${batchIdx + 1}/${batches.length}`);

    const positionsText = batch.map((p, i) => 
      `POSITION ${i + 1}: "${p.claim}"`
    ).join('\n\n');

    // Build the example format based on requirements
    let exampleFormat = '';
    if (format.startWithQuotedClaim) {
      if (format.sentenceCount) {
        exampleFormat = `---DEFENSE 1---
"[Exact claim text reproduced here]"

[Exactly ${format.sentenceCount} sentences of defense. Sentence one. Sentence two. ${format.sentenceCount > 2 ? `... Sentence ${format.sentenceCount}.` : ''}]

---DEFENSE 2---
"[Exact claim text reproduced here]"

[Exactly ${format.sentenceCount} sentences of defense.]`;
      } else {
        exampleFormat = `---DEFENSE 1---
"[Exact claim text reproduced here]"

[Defense content here...]

---DEFENSE 2---
"[Exact claim text reproduced here]"

[Defense content here...]`;
      }
    } else {
      exampleFormat = `---DEFENSE 1---
[Defense of position 1]

---DEFENSE 2---
[Defense of position 2]`;
    }

    const systemPrompt = `You are a skilled philosophical advocate. For each position, provide a rigorous defense.

CRITICAL FORMAT REQUIREMENTS:
${formatInstructions}

${format.startWithQuotedClaim ? `
MANDATORY: Each defense MUST begin with the EXACT claim text in quotation marks, reproduced VERBATIM.
Do NOT paraphrase. Do NOT summarize. Copy the claim EXACTLY as provided.
` : ''}

${format.sentenceCount ? `
MANDATORY: Each defense must contain EXACTLY ${format.sentenceCount} sentences.
- Count your sentences carefully before submitting
- A sentence ends with a period, question mark, or exclamation point
- Do not write ${format.sentenceCount - 1} sentences. Do not write ${format.sentenceCount + 1} sentences.
- Write EXACTLY ${format.sentenceCount} sentences.
` : ''}

CONTENT REQUIREMENTS:
1. Provide the strongest argument FOR this position
2. Be intellectually rigorous, not generic
3. Anticipate and rebut the most obvious objection when space permits

FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
${exampleFormat}

Continue for all positions.`;

    const response = await anthropic.messages.create({
      model: "claude-sonnet-4-20250514",
      max_tokens: 8000,
      system: systemPrompt,
      messages: [{
        role: "user",
        content: `Generate defenses for these ${batch.length} positions. ${format.startWithQuotedClaim ? 'Remember: START each defense with the EXACT claim in quotes.' : ''} ${format.sentenceCount ? `Remember: EXACTLY ${format.sentenceCount} sentences each.` : ''}\n\n${positionsText}`
      }]
    });

    const responseText = (response.content[0] as any).text;
    const defenseBlocks = responseText.split(/---DEFENSE \d+---/).filter((s: string) => s.trim().length > 0);

    for (let i = 0; i < batch.length; i++) {
      let defense = defenseBlocks[i]?.trim() || '';

      // FIX #6: Post-process to ensure claim is quoted at start if required
      if (format.startWithQuotedClaim && defense) {
        const claim = batch[i].claim;
        const startsWithQuote = defense.startsWith('"');

        if (!startsWithQuote) {
          // Prepend the quoted claim
          defense = `"${claim}"\n\n${defense}`;
        } else {
          // Verify the quote matches the claim (or is close)
          const firstQuoteEnd = defense.indexOf('"', 1);
          if (firstQuoteEnd > 0) {
            const quotedText = defense.substring(1, firstQuoteEnd);
            // If the quoted text is significantly different, replace it
            if (quotedText.length < claim.length * 0.5 || quotedText.length > claim.length * 1.5) {
              const afterQuote = defense.substring(firstQuoteEnd + 1).trim();
              defense = `"${claim}"\n\n${afterQuote}`;
            }
          }
        }
      }

      defenses.push({
        position: batch[i],
        defense: defense || 'Defense generation failed for this position.'
      });
    }

    if (batchIdx < batches.length - 1) {
      await new Promise(resolve => setTimeout(resolve, 500));
    }
  }

  return defenses;
}

export async function processPositionList(
  text: string,
  customInstructions: string = '',
  onProgress?: (stage: string, current: number, total: number) => void
): Promise<PositionListResult> {
  console.log('[POSITION-LIST] Starting position list processing');
  console.log(`[POSITION-LIST] Custom instructions: ${customInstructions || 'None'}`);

  if (!process.env.ANTHROPIC_API_KEY) {
    return {
      success: false,
      output: '',
      positionsProcessed: 0,
      positionsSelected: 0,
      totalPositions: 0,
      error: 'Anthropic API key not configured'
    };
  }

  try {
    onProgress?.('Parsing positions...', 0, 4);
    const positions = parsePositions(text);
    console.log(`[POSITION-LIST] Parsed ${positions.length} positions`);

    if (positions.length === 0) {
      return {
        success: false,
        output: '',
        positionsProcessed: 0,
        positionsSelected: 0,
        totalPositions: 0,
        error: 'No valid positions found in input'
      };
    }

    // FIX #7: Parse format requirements BEFORE determining selection count
    const format = parseDefenseFormat(customInstructions);

    // FIX #8: Pass total positions to parseSelectionCount
    const targetCount = parseSelectionCount(customInstructions, positions.length);
    const effectiveTarget = Math.min(targetCount, positions.length);
    console.log(`[POSITION-LIST] Targeting ${effectiveTarget} positions (requested: ${targetCount}, total: ${positions.length})`);

    onProgress?.('Ranking positions by significance...', 1, 4);
    const rankedPositions = await rankPositions(positions, customInstructions, effectiveTarget);
    console.log(`[POSITION-LIST] Selected ${rankedPositions.length} positions`);

    onProgress?.('Generating defenses...', 2, 4);
    // FIX #9: Pass format to generateDefenses
    const defenses = await generateDefenses(rankedPositions, customInstructions, format);
    console.log(`[POSITION-LIST] Generated ${defenses.length} defenses`);

    onProgress?.('Formatting output...', 3, 4);

    // FIX #10: Cleaner output format that respects the structure
    let output = `${'═'.repeat(55)}
POSITION DEFENSE ANALYSIS
${'═'.repeat(55)}
Total Positions Submitted: ${positions.length}
Positions Selected for Defense: ${rankedPositions.length}
Selection Criterion: ${customInstructions || 'All positions processed'}
${'═'.repeat(55)}

`;

    const byCategory: Record<string, PositionDefense[]> = {};
    for (const def of defenses) {
      const cat = def.position.category;
      if (!byCategory[cat]) byCategory[cat] = [];
      byCategory[cat].push(def);
    }

    for (const category of Object.keys(byCategory)) {
      const categoryDefenses = byCategory[category];
      output += `\n${'─'.repeat(55)}
CATEGORY: ${category.toUpperCase()}
${'─'.repeat(55)}\n\n`;

      for (const def of categoryDefenses) {
        // FIX #11: Simplified output - defense already contains quoted claim if required
        output += `POSITION (Score: ${def.position.significanceScore}/100):
"${def.position.claim}"

DEFENSE:
${def.defense}

${'─'.repeat(40)}

`;
      }
    }

    output += `\n${'═'.repeat(55)}
SUMMARY
${'═'.repeat(55)}
Categories covered: ${Object.keys(byCategory).length}
Average significance score: ${Math.round(rankedPositions.reduce((s, p) => s + p.significanceScore, 0) / rankedPositions.length)}
${'═'.repeat(55)}`;

    onProgress?.('Complete', 4, 4);

    return {
      success: true,
      output,
      positionsProcessed: defenses.length,
      positionsSelected: rankedPositions.length,
      totalPositions: positions.length
    };

  } catch (error: any) {
    console.error('[POSITION-LIST] Error:', error);
    return {
      success: false,
      output: '',
      positionsProcessed: 0,
      positionsSelected: 0,
      totalPositions: 0,
      error: error.message || 'Position list processing failed'
    };
  }
}