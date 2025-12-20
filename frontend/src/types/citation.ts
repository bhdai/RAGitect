/**
 * Citation types for AI SDK streaming integration
 *
 * Story 3.2.B: Streaming LLM Responses with Citations
 *
 * These types match the backend Citation model from ragitect/api/schemas/chat.py
 * and are compatible with AI SDK source-document parts.
 */

/**
 * RAGitect-specific citation metadata embedded in providerMetadata.ragitect
 */
export interface RagitectCitationMetadata {
  /** Index of chunk within the document */
  chunkIndex: number;
  /** Relevance score 0-1 from vector search/reranking */
  similarity: number;
  /** Full chunk content for tooltip preview */
  preview: string;
}

/**
 * Citation data structure for rendering inline citations
 *
 * Mapped from AI SDK source-document parts to a flat structure
 * for easier consumption by MessageWithCitations component.
 */
export interface CitationData {
  /** Unique citation ID (cite-0, cite-1, ...) */
  id: string;
  /** Source document filename */
  title: string;
  /** MIME type of source (text/plain for text chunks) */
  mediaType: string;
  /** Chunk index within document */
  chunkIndex: number;
  /** Relevance score 0-1 */
  similarity: number;
  /** Full chunk content for preview */
  preview: string;
}

/**
 * Map of citation ID to citation data for quick lookup
 *
 * Used by MessageWithCitations to resolve [N] markers to citation data.
 */
export type CitationMap = Record<string, CitationData>;

/**
 * AI SDK source-document part structure
 *
 * This matches what the AI SDK useChat hook provides in message.parts
 * for source-document type parts.
 */
export interface SourceDocumentPart {
  type: 'source-document';
  sourceId: string;
  mediaType?: string;
  title: string;
  providerMetadata?: {
    ragitect?: RagitectCitationMetadata;
  };
  [key: string]: unknown;
}

/**
 * Type guard to check if a message part is a source-document
 */
export function isSourceDocumentPart(
  part: { type: string;[key: string]: unknown }
): part is SourceDocumentPart {
  return part.type === 'source-document';
}

/**
 * Extract CitationData from a source-document part
 *
 * @param part - AI SDK source-document part
 * @returns CitationData for use in MessageWithCitations
 */
export function extractCitationData(part: SourceDocumentPart): CitationData {
  const ragitectMeta = part.providerMetadata?.ragitect;
  return {
    id: part.sourceId,
    title: part.title,
    mediaType: part.mediaType || 'text/plain',
    chunkIndex: ragitectMeta?.chunkIndex ?? 0,
    similarity: ragitectMeta?.similarity ?? 0,
    preview: ragitectMeta?.preview ?? '',
  };
}

/**
 * Build a CitationMap from an array of source-document parts
 *
 * @param parts - Array of AI SDK message parts
 * @returns CitationMap for use in MessageWithCitations
 */
export function buildCitationMap(
  parts: Array<{ type: string;[key: string]: unknown }>
): CitationMap {
  const map: CitationMap = {};
  for (const part of parts) {
    if (isSourceDocumentPart(part)) {
      map[part.sourceId] = extractCitationData(part);
    }
  }
  return map;
}
