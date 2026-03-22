const path = require("path");

const projectRoot = __dirname;
const venvBin = path.join(projectRoot, ".venv", "bin");

module.exports = {
  apps : [{
    name: "eliashberg-web",
    script: "/Applications/quarto/bin/quarto",
    interpreter: "none",
    args: "preview --port 4200 --host 0.0.0.0",
    cwd: path.join(projectRoot, "web"),
    // 环境变量
    env: {
      JULIA_PROJECT: projectRoot,
      PYTHONPATH: projectRoot,
      QUARTO_PYTHON: path.join(venvBin, "python"),
      PATH: `${venvBin}:${process.env.PATH}`
    },
    watch: false,
    autorestart: true,
  }]
}
