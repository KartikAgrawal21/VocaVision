package com.vocavision.app

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.RectF
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/** One detection; box coordinates are normalised (0..1) against the upright source image. */
class Det(val cls: Int, val conf: Float, val x1: Float, val y1: Float, val x2: Float, val y2: Float)

/** Runs the custom VocaVision YOLOv8 model (exported to ONNX) on the phone with ONNX Runtime. */
class Detector(context: Context) {

    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    // Reused buffers: allocating a tensor and bitmap every frame would hurt FPS.
    private var bufW = 0
    private var bufH = 0
    private var canvasBmp: Bitmap? = null
    private var pixels = IntArray(0)
    private var input: FloatBuffer? = null

    private val paint = Paint(Paint.FILTER_BITMAP_FLAG)

    init {
        val bytes = context.assets.open(MODEL_FILE).use { it.readBytes() }
        val opts = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(THREADS)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        session = env.createSession(bytes, opts)
    }

    /** [size] is the long side of the network input (the short side is scaled and padded to /32). */
    fun detect(src: Bitmap, size: Int): List<Det> {
        val scale = size.toFloat() / max(src.width, src.height)
        val nw = (src.width * scale).roundToInt().coerceAtLeast(1)
        val nh = (src.height * scale).roundToInt().coerceAtLeast(1)
        val w = ceil32(nw)
        val h = ceil32(nh)
        val dx = (w - nw) / 2f
        val dy = (h - nh) / 2f

        ensureBuffers(w, h)
        val bmp = canvasBmp!!
        val canvas = Canvas(bmp)
        canvas.drawColor(PAD_GREY)
        canvas.drawBitmap(src, null, RectF(dx, dy, dx + nw, dy + nh), paint)
        bmp.getPixels(pixels, 0, w, 0, 0, w, h)

        // ARGB ints -> planar RGB floats in 0..1 (the layout YOLO expects).
        val buf = input!!
        val plane = w * h
        for (i in 0 until plane) {
            val p = pixels[i]
            buf.put(i, ((p shr 16) and 0xFF) / 255f)
            buf.put(plane + i, ((p shr 8) and 0xFF) / 255f)
            buf.put(2 * plane + i, (p and 0xFF) / 255f)
        }
        buf.rewind()

        OnnxTensor.createTensor(env, buf, longArrayOf(1, 3, h.toLong(), w.toLong())).use { tensor ->
            session.run(mapOf(session.inputNames.first() to tensor)).use { out ->
                @Suppress("UNCHECKED_CAST")
                val raw = (out[0].value as Array<Array<FloatArray>>)[0]  // [4 + classes][anchors]
                return decode(raw, dx, dy, nw, nh)
            }
        }
    }

    private fun decode(raw: Array<FloatArray>, dx: Float, dy: Float, nw: Int, nh: Int): List<Det> {
        val anchors = raw[0].size
        val cands = ArrayList<Det>()
        for (i in 0 until anchors) {
            var best = -1
            var bestScore = 0f
            for (c in 0 until Labels.COUNT) {
                val s = raw[4 + c][i]
                if (s > bestScore) { bestScore = s; best = c }
            }
            if (best < 0 || bestScore < Labels.threshold(best)) continue

            val cx = raw[0][i]; val cy = raw[1][i]
            val bw = raw[2][i]; val bh = raw[3][i]
            val x1 = ((cx - bw / 2 - dx) / nw).coerceIn(0f, 1f)
            val y1 = ((cy - bh / 2 - dy) / nh).coerceIn(0f, 1f)
            val x2 = ((cx + bw / 2 - dx) / nw).coerceIn(0f, 1f)
            val y2 = ((cy + bh / 2 - dy) / nh).coerceIn(0f, 1f)
            if (x2 - x1 < 0.01f || y2 - y1 < 0.01f) continue
            cands.add(Det(best, bestScore, x1, y1, x2, y2))
        }
        return nms(cands)
    }

    /** Class-aware non-maximum suppression, same idea as the desktop (Ultralytics) pipeline. */
    private fun nms(cands: List<Det>): List<Det> {
        val sorted = cands.sortedByDescending { it.conf }
        val kept = ArrayList<Det>()
        for (d in sorted) {
            if (kept.size >= MAX_DETECTIONS) break
            // Same class: usual NMS. Same group (both clothing or both food) but different class,
            // e.g. "shirt" and "T-shirt" on one garment: keep only the more confident one.
            // A shirt and trousers barely overlap, so both survive.
            val dup = kept.any {
                (it.cls == d.cls && iou(it, d) > IOU_LIMIT) ||
                    (Labels.isFood(it.cls) == Labels.isFood(d.cls) && iou(it, d) > GROUP_IOU_LIMIT)
            }
            if (!dup) kept.add(d)
        }
        return kept
    }

    private fun iou(a: Det, b: Det): Float {
        val ix = max(0f, min(a.x2, b.x2) - max(a.x1, b.x1))
        val iy = max(0f, min(a.y2, b.y2) - max(a.y1, b.y1))
        val inter = ix * iy
        val union = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter
        return if (union <= 0f) 0f else inter / union
    }

    private fun ensureBuffers(w: Int, h: Int) {
        if (w == bufW && h == bufH) return
        bufW = w; bufH = h
        canvasBmp?.recycle()
        canvasBmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        pixels = IntArray(w * h)
        input = ByteBuffer.allocateDirect(4 * 3 * w * h).order(ByteOrder.nativeOrder()).asFloatBuffer()
    }

    private fun ceil32(v: Int) = ((v + 31) / 32) * 32

    companion object {
        private const val MODEL_FILE = "vocavision_yolo.onnx"
        private const val THREADS = 4
        private const val PAD_GREY = 0xFF727272.toInt()  // 114, the Ultralytics letterbox colour
        private const val IOU_LIMIT = 0.7f
        private const val GROUP_IOU_LIMIT = 0.75f
        private const val MAX_DETECTIONS = 20
    }
}
