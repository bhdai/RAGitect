/**
 * Tests for citation utility functions
 *
 * AC3/AC5: stripCitations removes [cite: N] markers for clean copy
 */

import { describe, it, expect } from 'vitest';
import { stripCitations } from '../citations';

describe('stripCitations', () => {
  it('removes single citation marker', () => {
    const input = 'Python is powerful [cite: 1].';
    const expected = 'Python is powerful.';
    expect(stripCitations(input)).toBe(expected);
  });

  it('removes multiple citation markers', () => {
    const input = 'Hello [cite: 1] world [cite: 2].';
    const expected = 'Hello world.';
    expect(stripCitations(input)).toBe(expected);
  });

  it('removes consecutive citation markers', () => {
    const input = 'This is supported by evidence [cite: 1][cite: 2][cite: 3].';
    const expected = 'This is supported by evidence.';
    expect(stripCitations(input)).toBe(expected);
  });

  it('handles citation with no space after colon', () => {
    const input = 'Python[cite:1] is great.';
    const expected = 'Python is great.';
    expect(stripCitations(input)).toBe(expected);
  });

  it('handles citation with extra space after colon', () => {
    const input = 'Python[cite:  1] is great.';
    const expected = 'Python is great.';
    expect(stripCitations(input)).toBe(expected);
  });

  it('returns original text when no citations', () => {
    const input = 'No citations here.';
    expect(stripCitations(input)).toBe(input);
  });

  it('handles empty string', () => {
    expect(stripCitations('')).toBe('');
  });

  it('does NOT strip old [N] format (intentional)', () => {
    // Old format should NOT be stripped - this is intentional
    // If backend sends old format, we want user to see it's wrong
    const input = 'Python is powerful [0].';
    expect(stripCitations(input)).toBe('Python is powerful [0].');
  });

  it('collapses multiple spaces to single space', () => {
    const input = 'Python   [cite: 1]   is great.';
    const expected = 'Python is great.';
    expect(stripCitations(input)).toBe(expected);
  });

  it('trims leading and trailing whitespace', () => {
    const input = '  Python is great [cite: 1].  ';
    const expected = 'Python is great.';
    expect(stripCitations(input)).toBe(expected);
  });

  it('handles multiline text', () => {
    const input = 'Python is powerful [cite: 1].\n\nIt supports multiple paradigms [cite: 2].';
    const expected = 'Python is powerful.\n\nIt supports multiple paradigms.';
    expect(stripCitations(input)).toBe(expected);
  });

  it('handles large citation indices', () => {
    const input = 'Reference [cite: 123] here.';
    const expected = 'Reference here.';
    expect(stripCitations(input)).toBe(expected);
  });
});
