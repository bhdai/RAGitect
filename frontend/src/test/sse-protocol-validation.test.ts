/**
 * SSE Protocol Validation Tests
 * 
 * Story 3.2.B: Streaming LLM Responses with Citations - Task 0.5
 * 
 * CRITICAL: Validates that our UI Message Stream Protocol v1 implementation
 * correctly handles source-document parts for citations.
 * 
 * These tests verify the SSE event format and parsing behavior
 * before we integrate citations into the ChatPanel.
 */

import { describe, it, expect } from 'vitest';

/**
 * Simulates parsing a raw SSE event line into structured data.
 * Mirrors what the AI SDK does internally.
 */
function parseSSEEvent(line: string): unknown {
  if (!line.startsWith('data: ')) {
    return null;
  }
  const jsonStr = line.slice(6); // Remove "data: " prefix
  try {
    return JSON.parse(jsonStr);
  } catch {
    return null;
  }
}

/**
 * Source document part type matching AI SDK SourceDocumentUIPart
 */
interface SourceDocumentPart {
  type: 'source-document';
  sourceId: string;
  mediaType: string;
  title: string;
  filename?: string;
  providerMetadata?: {
    ragitect?: {
      chunkIndex: number;
      similarity: number;
      preview: string;
    };
  };
}

/**
 * Text delta part type
 */
interface TextDeltaPart {
  type: 'text-delta';
  id: string;
  delta: string;
}

describe('UI Message Stream Protocol v1 Validation', () => {
  describe('SSE Event Parsing', () => {
    it('parses text-delta events correctly', () => {
      const event = 'data: {"type":"text-delta","id":"text-1","delta":"Hello"}';
      const parsed = parseSSEEvent(event) as TextDeltaPart;
      
      expect(parsed).toEqual({
        type: 'text-delta',
        id: 'text-1',
        delta: 'Hello',
      });
    });

    it('parses source-document events correctly', () => {
      const event = 'data: {"type":"source-document","sourceId":"cite-0","mediaType":"text/plain","title":"intro.pdf","providerMetadata":{"ragitect":{"chunkIndex":0,"similarity":0.95,"preview":"Python is..."}}}';
      const parsed = parseSSEEvent(event) as SourceDocumentPart;
      
      expect(parsed).toEqual({
        type: 'source-document',
        sourceId: 'cite-0',
        mediaType: 'text/plain',
        title: 'intro.pdf',
        providerMetadata: {
          ragitect: {
            chunkIndex: 0,
            similarity: 0.95,
            preview: 'Python is...',
          },
        },
      });
    });

    it('parses start/finish events correctly', () => {
      const startEvent = 'data: {"type":"start","messageId":"msg-1"}';
      const finishEvent = 'data: {"type":"finish","finishReason":"stop"}';
      
      expect(parseSSEEvent(startEvent)).toEqual({
        type: 'start',
        messageId: 'msg-1',
      });
      
      expect(parseSSEEvent(finishEvent)).toEqual({
        type: 'finish',
        finishReason: 'stop',
      });
    });
  });

  describe('Interleaved Text and Citation Stream', () => {
    it('correctly parses a stream with interleaved text and source-document parts', () => {
      // Simulate a complete SSE stream with citations
      const mockStream = [
        'data: {"type":"start","messageId":"msg-1"}',
        'data: {"type":"text-start","id":"text-1"}',
        'data: {"type":"text-delta","id":"text-1","delta":"Python is a "}',
        'data: {"type":"text-delta","id":"text-1","delta":"powerful"}',
        'data: {"type":"source-document","sourceId":"cite-0","mediaType":"text/plain","title":"intro.pdf","providerMetadata":{"ragitect":{"chunkIndex":0,"similarity":0.95,"preview":"Python is..."}}}',
        'data: {"type":"text-delta","id":"text-1","delta":"[0]"}',
        'data: {"type":"text-delta","id":"text-1","delta":" language"}',
        'data: {"type":"text-end","id":"text-1"}',
        'data: {"type":"finish","finishReason":"stop"}',
      ];
      
      // Parse all events
      const events = mockStream.map(parseSSEEvent).filter(Boolean);
      
      // Extract text deltas
      const textDeltas = events.filter(
        (e): e is TextDeltaPart => (e as TextDeltaPart).type === 'text-delta'
      );
      const fullText = textDeltas.map((t) => t.delta).join('');
      
      // Extract source documents
      const sourceDocs = events.filter(
        (e): e is SourceDocumentPart => (e as SourceDocumentPart).type === 'source-document'
      );
      
      // Verify text content
      expect(fullText).toBe('Python is a powerful[0] language');
      
      // Verify citation metadata
      expect(sourceDocs).toHaveLength(1);
      expect(sourceDocs[0].sourceId).toBe('cite-0');
      expect(sourceDocs[0].title).toBe('intro.pdf');
      expect(sourceDocs[0].providerMetadata?.ragitect?.similarity).toBe(0.95);
    });

    it('handles multiple citations in a single stream', () => {
      const mockStream = [
        'data: {"type":"start","messageId":"msg-1"}',
        'data: {"type":"text-start","id":"text-1"}',
        'data: {"type":"text-delta","id":"text-1","delta":"Python is powerful"}',
        'data: {"type":"source-document","sourceId":"cite-0","mediaType":"text/plain","title":"intro.pdf","providerMetadata":{"ragitect":{"chunkIndex":0,"similarity":0.95,"preview":"Power..."}}}',
        'data: {"type":"text-delta","id":"text-1","delta":"[0]"}',
        'data: {"type":"text-delta","id":"text-1","delta":" and versatile"}',
        'data: {"type":"source-document","sourceId":"cite-1","mediaType":"text/plain","title":"advanced.pdf","providerMetadata":{"ragitect":{"chunkIndex":3,"similarity":0.87,"preview":"Versatile..."}}}',
        'data: {"type":"text-delta","id":"text-1","delta":"[1]"}',
        'data: {"type":"text-delta","id":"text-1","delta":"."}',
        'data: {"type":"text-end","id":"text-1"}',
        'data: {"type":"finish","finishReason":"stop"}',
      ];
      
      const events = mockStream.map(parseSSEEvent).filter(Boolean);
      
      const sourceDocs = events.filter(
        (e): e is SourceDocumentPart => (e as SourceDocumentPart).type === 'source-document'
      );
      
      expect(sourceDocs).toHaveLength(2);
      expect(sourceDocs[0].sourceId).toBe('cite-0');
      expect(sourceDocs[0].title).toBe('intro.pdf');
      expect(sourceDocs[1].sourceId).toBe('cite-1');
      expect(sourceDocs[1].title).toBe('advanced.pdf');
    });
  });

  describe('Zero Citation Stream', () => {
    it('handles messages without any source-document parts', () => {
      const mockStream = [
        'data: {"type":"start","messageId":"msg-1"}',
        'data: {"type":"text-start","id":"text-1"}',
        'data: {"type":"text-delta","id":"text-1","delta":"2 plus 2 equals 4."}',
        'data: {"type":"text-end","id":"text-1"}',
        'data: {"type":"finish","finishReason":"stop"}',
      ];
      
      const events = mockStream.map(parseSSEEvent).filter(Boolean);
      
      const sourceDocs = events.filter(
        (e): e is SourceDocumentPart => (e as SourceDocumentPart).type === 'source-document'
      );
      
      // No source documents expected
      expect(sourceDocs).toHaveLength(0);
      
      // Text should still be present
      const textDeltas = events.filter(
        (e): e is TextDeltaPart => (e as TextDeltaPart).type === 'text-delta'
      );
      expect(textDeltas[0].delta).toBe('2 plus 2 equals 4.');
    });
  });

  describe('Citation Marker Parsing', () => {
    it('extracts citation indices from text content', () => {
      const text = 'Python is powerful[0] and versatile[1].';
      
      // Pattern to match [N] markers
      const pattern = /\[(\d+)\]/g;
      const matches: number[] = [];
      let match;
      
      while ((match = pattern.exec(text)) !== null) {
        matches.push(parseInt(match[1], 10));
      }
      
      expect(matches).toEqual([0, 1]);
    });

    it('splits text on citation markers for rendering', () => {
      const text = 'Python is powerful[0] and versatile[1].';
      
      // Split preserving the markers
      const parts = text.split(/(\[\d+\])/);
      
      expect(parts).toEqual([
        'Python is powerful',
        '[0]',
        ' and versatile',
        '[1]',
        '.',
      ]);
    });

    it('handles text without any citation markers', () => {
      const text = 'Python is a programming language.';
      
      const parts = text.split(/(\[\d+\])/);
      
      expect(parts).toEqual(['Python is a programming language.']);
    });

    it('handles consecutive citation markers', () => {
      const text = 'Python[0][1] is great.';
      
      const parts = text.split(/(\[\d+\])/);
      
      expect(parts).toEqual(['Python', '[0]', '', '[1]', ' is great.']);
    });
  });

  describe('Citation Data Mapping', () => {
    it('maps source-document parts to CitationData format', () => {
      const sourceDoc: SourceDocumentPart = {
        type: 'source-document',
        sourceId: 'cite-0',
        mediaType: 'text/plain',
        title: 'intro.pdf',
        providerMetadata: {
          ragitect: {
            chunkIndex: 5,
            similarity: 0.92,
            preview: 'Python is a powerful programming language...',
          },
        },
      };
      
      // Map to our CitationData format
      const citationData = {
        type: 'citation' as const,
        id: sourceDoc.sourceId,
        docName: sourceDoc.title,
        chunkIdx: sourceDoc.providerMetadata?.ragitect?.chunkIndex ?? 0,
        similarity: sourceDoc.providerMetadata?.ragitect?.similarity ?? 0,
        contentPreview: sourceDoc.providerMetadata?.ragitect?.preview ?? '',
      };
      
      expect(citationData).toEqual({
        type: 'citation',
        id: 'cite-0',
        docName: 'intro.pdf',
        chunkIdx: 5,
        similarity: 0.92,
        contentPreview: 'Python is a powerful programming language...',
      });
    });

    it('handles missing providerMetadata gracefully', () => {
      const sourceDoc: SourceDocumentPart = {
        type: 'source-document',
        sourceId: 'cite-0',
        mediaType: 'text/plain',
        title: 'intro.pdf',
        // No providerMetadata
      };
      
      const citationData = {
        type: 'citation' as const,
        id: sourceDoc.sourceId,
        docName: sourceDoc.title,
        chunkIdx: sourceDoc.providerMetadata?.ragitect?.chunkIndex ?? 0,
        similarity: sourceDoc.providerMetadata?.ragitect?.similarity ?? 0,
        contentPreview: sourceDoc.providerMetadata?.ragitect?.preview ?? '',
      };
      
      // Should have defaults
      expect(citationData.chunkIdx).toBe(0);
      expect(citationData.similarity).toBe(0);
      expect(citationData.contentPreview).toBe('');
    });
  });
});
