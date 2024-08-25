# Hand Tracking Project

This project demonstrates real-time hand tracking using MediaPipe and TypeScript, bundled with Webpack.

## Prerequisites

- Node.js (v14 or later recommended)
- npm (comes with Node.js)

## Installation

1. Clone the repository and install dependencies:
   ```
   cd hand-tracking-project
   npm install
   ```

## Development

To run the project in development mode with hot-reloading:

```
npm start
```

This will start a development server and open the application in your default web browser.

## Building for Production

To create a production build:

```
npm run build
```

This will generate production-ready files in the `dist` directory.

## Project Structure

```
.
├── src/
│   ├── index.html
│   └── script.ts
├── package.json
├── tsconfig.json
├── webpack.config.js
├── .gitignore
└── README.md
```

- `src/`: Contains the source code
  - `index.html`: The main HTML file
  - `script.ts`: The main TypeScript file containing the hand tracking logic
- `package.json`: Defines npm package dependencies and scripts
- `tsconfig.json`: TypeScript compiler configuration
- `webpack.config.js`: Webpack configuration file

## Technologies Used

- TypeScript
- MediaPipe
- Webpack
- HTML5 Canvas

## License

[MIT License](https://opensource.org/licenses/MIT)
