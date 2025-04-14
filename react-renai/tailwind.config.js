// module.exports = {
//     content: [
//       "./index.html",
//       "./src/**/*.{js,jsx,ts,tsx}",
//     ],
//     theme: {
//       extend: {
//         // Пример добавления кастомной анимации
//         animation: {
//           'spin-slow': 'spin 8s linear infinite',
//           'fadeIn': 'fadeIn 1s ease-out forwards',
//         },
//         keyframes: {
//           fadeIn: {
//             '0%': { opacity: 0 },
//             '100%': { opacity: 1 },
//           },
//         },
//       },
//     },
//     plugins: [],
//   }

module.exports = {
  content: ["./index.html", "./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      animation: {
        gradientShift: "gradientShift 8s ease infinite",
        slideIn: "slideIn 1s ease-out forwards",
        fadeIn: "fadeIn 1s ease-out forwards",
      },
      keyframes: {
        gradientShift: {
          "0%, 100%": { backgroundSize: "200% 200%", backgroundPosition: "0% 50%" },
          "50%": { backgroundSize: "200% 200%", backgroundPosition: "100% 50%" },
        },
        slideIn: {
          "0%": { transform: "translateY(20px)", opacity: 0 },
          "100%": { transform: "translateY(0)", opacity: 1 },
        },
        fadeIn: {
          "0%": { opacity: 0 },
          "100%": { opacity: 1 },
        },
      },
    },
  },
  plugins: [],
};