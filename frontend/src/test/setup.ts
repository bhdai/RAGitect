import '@testing-library/jest-dom'

// Polyfill for Radix UI components (hasPointerCapture not implemented in JSDOM)
// See: https://github.com/radix-ui/primitives/issues/1207
if (!Element.prototype.hasPointerCapture) {
  Element.prototype.hasPointerCapture = function() {
    return false;
  };
}

if (!Element.prototype.setPointerCapture) {
  Element.prototype.setPointerCapture = function() {};
}

if (!Element.prototype.releasePointerCapture) {
  Element.prototype.releasePointerCapture = function() {};
}

// Polyfill for scrollIntoView (needed by Radix UI Select)
if (!Element.prototype.scrollIntoView) {
  Element.prototype.scrollIntoView = function() {};
}
