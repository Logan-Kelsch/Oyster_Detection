import labels from "./labelsO.json";

/**
 * Bakes annotations directly into image data
 * @param {HTMLImageElement|HTMLVideoElement} source - Original media element
 * @param {Array} boxes_data - Detection boxes
 * @param {Array} scores_data - Confidence scores
 * @param {Array} classes_data - Class indices
 * @param {Array[Number]} ratios - Scaling ratios [xRatio, yRatio]
 * @returns {HTMLCanvasElement} - Canvas with baked-in annotations
 */
export const bakeAnnotations = (source, boxes_data, scores_data, classes_data, ratios) => {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    // Set canvas to original media dimensions
    canvas.width = source.videoWidth || source.width;
    canvas.height = source.videoHeight || source.height;

    // 1. Draw original image onto canvas
    ctx.drawImage(source, 0, 0, canvas.width, canvas.height);

    const colors = new Colors();
    const font = `${Math.max(canvas.width / 50, 12)}px Arial`;
    ctx.font = font;
    ctx.textBaseline = "top";

    for (let i = 0; i < scores_data.length; ++i) {
        const klass = labels[classes_data[i]];
        const color = colors.get(classes_data[i]);
        const score = (scores_data[i] * 100).toFixed(1);

        // Scale box coordinates to original dimensions
        let [y1, x1, y2, x2] = boxes_data.slice(i * 4, (i + 1) * 4).map(
            (val, idx) => idx % 2 === 0 ? val * ratios[1] : val * ratios[0]
        );

        const width = x2 - x1;
        const height = y2 - y1;

        // 2. Draw directly on the image
        ctx.fillStyle = Colors.hexToRgba(color, 0.2);
        ctx.fillRect(x1, y1, width, height);

        ctx.strokeStyle = color;
        ctx.lineWidth = Math.max(canvas.width / 200, 1.5);
        ctx.strokeRect(x1, y1, width, height);

        // Label background
        ctx.fillStyle = color;
        const text = `${klass} - ${score}%`;
        const textWidth = ctx.measureText(text).width;
        const textHeight = parseInt(font, 10);
        const yText = Math.max(0, y1 - textHeight);

        ctx.fillRect(x1 - 1, yText, textWidth + 2, textHeight);
        ctx.fillStyle = "#ffffff";
        ctx.fillText(text, x1, yText);
    }

    return canvas; // Return canvas with baked annotations
};