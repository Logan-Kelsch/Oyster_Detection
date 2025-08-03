import { createFFmpeg, fetchFile } from "@ffmpeg/ffmpeg";

const ffmpeg = createFFmpeg({ log: true });

export const encodeWithFFmpeg = async (frames, width, height, fps) => {
    if (!ffmpeg.isLoaded()) await ffmpeg.load();

    for (let i = 0; i < frames.length; i++) {
        const name = `frame_${String(i).padStart(4, "0")}.jpg`;
        ffmpeg.FS("writeFile", name, await fetchFile(frames[i].blob));
    }

    await ffmpeg.run(
        "-framerate", String(fps),
        "-i", "frame_%04d.jpg",
        "-s", `${width}x${height}`,
        "-c:v", "libvpx",
        "-b:v", "1M",
        "output.webm"
    );

    const data = ffmpeg.FS("readFile", "output.webm");
    return new Blob([data.buffer], { type: "video/webm" });
};
