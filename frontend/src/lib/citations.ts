/**
 * Citation utility functions
 *
 * Provides utilities for working with citation markers in text.
 * ADR-3.4.1: Uses [cite: N] format throughout.
 */

/**
 * Strip all [cite: N] markers from text
 *
 * Used for copy functionality - citations are meaningful in the chat UI
 * but not when pasting text externally.
 *
 * @param text - Text with potential [cite: N] markers
 * @returns Clean text without citation markers
 *
 * @example
 * stripCitations("Hello [cite: 0] world [cite: 1].") // "Hello world."
 */
export function stripCitations(text: string): string {
  return text
    // Remove [cite: N] markers (with flexible spacing)
    .replace(/\[cite:\s*\d+\]/g, '')
    // Remove space before punctuation (from removed citations)
    .replace(/ +([.!?,;:])/g, '$1')
    // Collapse multiple spaces (but not newlines) to single space
    .replace(/ {2,}/g, ' ')
    // Trim whitespace
    .trim();
}
