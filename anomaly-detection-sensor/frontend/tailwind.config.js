/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        // Instrument-console palette grounded in validated-system signage.
        ink: '#0A0F1A', // deep slate background
        panel: '#111A28', // surface
        'panel-2': '#16212F', // raised surface
        line: '#22304A', // hairline borders
        'line-soft': '#1A2536',
        normal: '#2DD4BF', // teal = in-control / validated
        warn: '#F5A524', // amber = warning (GMP caution)
        alarm: '#F43F5E', // rose = alarm (GMP critical)
        accent: '#38BDF8', // sky = highlight / interactive
        ok: '#34D399',
        'text-hi': '#E6EDF6',
        'text-mid': '#9FB0C5',
        'text-lo': '#5E708A',
      },
      fontFamily: {
        sans: ['"IBM Plex Sans"', 'system-ui', 'sans-serif'],
        mono: ['"IBM Plex Mono"', 'ui-monospace', 'monospace'],
      },
      boxShadow: {
        panel: '0 1px 0 0 rgba(255,255,255,0.03) inset, 0 8px 24px -12px rgba(0,0,0,0.6)',
        glow: '0 0 0 1px rgba(56,189,248,0.4), 0 0 20px -4px rgba(56,189,248,0.35)',
      },
      keyframes: {
        'fade-up': {
          '0%': { opacity: '0', transform: 'translateY(6px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        'pulse-ring': {
          '0%': { opacity: '0.55', transform: 'scale(1)' },
          '70%': { opacity: '0', transform: 'scale(2.4)' },
          '100%': { opacity: '0', transform: 'scale(2.4)' },
        },
      },
      animation: {
        'fade-up': 'fade-up 0.4s ease-out both',
        'pulse-ring': 'pulse-ring 2.4s ease-out infinite',
      },
    },
  },
  plugins: [],
};
