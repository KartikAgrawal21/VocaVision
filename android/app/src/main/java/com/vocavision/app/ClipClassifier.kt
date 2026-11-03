package com.vocavision.app

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import org.json.JSONObject
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * CLIP (OpenAI, pre-trained) running on the phone. The image encoder is an int8 ONNX file;
 * the text side (labels and colour prompts) was embedded on the PC and ships as clip_text.json.
 * Same prompts and scoring as utils/clip_classifier.py on the desktop.
 */
class ClipClassifier(context: Context) {

    class Guess(val index: Int, val name: String, val kind: String, val conf: Float, val probs: FloatArray)

    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    private val names = ArrayList<String>()
    private val kinds = ArrayList<String>()
    private val labelEmb: Array<FloatArray>
    private val colours = ArrayList<String>()
    private val colourEmb: Array<FloatArray>

    private val input = ByteBuffer.allocateDirect(4 * 3 * SIDE * SIDE).order(ByteOrder.nativeOrder()).asFloatBuffer()
    private val px = IntArray(SIDE * SIDE)

    init {
        val json = JSONObject(context.assets.open("clip_text.json").bufferedReader().use { it.readText() })
        val labels = json.getJSONArray("labels")
        for (i in 0 until labels.length()) {
            val o = labels.getJSONObject(i)
            names.add(o.getString("name")); kinds.add(o.getString("kind"))
        }
        labelEmb = readMatrix(json.getJSONArray("label_emb"))
        val c = json.getJSONArray("colours")
        for (i in 0 until c.length()) colours.add(c.getString(i))
        colourEmb = readMatrix(json.getJSONArray("colour_emb"))

        val bytes = context.assets.open("clip_image.onnx").use { it.readBytes() }
        val opts = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(4)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        session = env.createSession(bytes, opts)
    }

    /** Scores the whole photo and its centre crop (where the item usually is) and averages them. */
    fun classify(src: Bitmap): Guess {
        val centre = cropOf(src, 0.2f, 0.2f, 0.8f, 0.95f)
        val probs = FloatArray(names.size)
        for (img in listOf(src, centre)) {
            val e = embed(img)
            val p = softmax(FloatArray(names.size) { 100f * dot(e, labelEmb[it]) })
            for (i in probs.indices) probs[i] += p[i] / 2f
        }
        var best = 0
        for (i in probs.indices) if (probs[i] > probs[best]) best = i
        return Guess(best, names[best], kinds[best], probs[best], probs)
    }

    fun name(i: Int) = names[i]
    fun kind(i: Int) = kinds[i]

    /** Colour of the region given as fractions of the image (left, top, right, bottom). */
    fun colourOf(src: Bitmap, left: Float, top: Float, right: Float, bottom: Float): String {
        val e = embed(cropOf(src, left, top, right, bottom))
        val p = FloatArray(colours.size) { 100f * dot(e, colourEmb[it]) }
        var best = 0
        for (i in p.indices) if (p[i] > p[best]) best = i
        return colours[best]
    }

    /** A sub-picture given as fractions of the image (left, top, right, bottom). */
    fun cropOf(src: Bitmap, l: Float, t: Float, r: Float, b: Float): Bitmap {
        val x = (l * src.width).toInt().coerceIn(0, src.width - 2)
        val y = (t * src.height).toInt().coerceIn(0, src.height - 2)
        val w = ((r - l) * src.width).toInt().coerceIn(2, src.width - x)
        val h = ((b - t) * src.height).toInt().coerceIn(2, src.height - y)
        return Bitmap.createBitmap(src, x, y, w, h)
    }

    /** Resize the short side to 224, centre-crop 224x224, normalise like CLIPProcessor, then encode. */
    private fun embed(src: Bitmap): FloatArray {
        val s = SIDE.toFloat() / min(src.width, src.height)
        val sw = max(SIDE, (src.width * s).toInt())
        val sh = max(SIDE, (src.height * s).toInt())
        val scaled = Bitmap.createScaledBitmap(src, sw, sh, true)
        val sq = Bitmap.createBitmap(scaled, (sw - SIDE) / 2, (sh - SIDE) / 2, SIDE, SIDE)
        sq.getPixels(px, 0, SIDE, 0, 0, SIDE, SIDE)
        val plane = SIDE * SIDE
        for (i in 0 until plane) {
            val p = px[i]
            input.put(i, (((p shr 16) and 0xFF) / 255f - MEAN[0]) / STD[0])
            input.put(plane + i, (((p shr 8) and 0xFF) / 255f - MEAN[1]) / STD[1])
            input.put(2 * plane + i, ((p and 0xFF) / 255f - MEAN[2]) / STD[2])
        }
        input.rewind()
        OnnxTensor.createTensor(env, input, longArrayOf(1, 3, SIDE.toLong(), SIDE.toLong())).use { t ->
            session.run(mapOf(session.inputNames.first() to t)).use { out ->
                @Suppress("UNCHECKED_CAST")
                val v = (out[0].value as Array<FloatArray>)[0].copyOf()
                var n = 0f
                for (x in v) n += x * x
                n = sqrt(n)
                for (i in v.indices) v[i] /= n
                return v
            }
        }
    }

    private fun softmax(x: FloatArray): FloatArray {
        val m = x.max()
        val e = FloatArray(x.size) { exp(x[it] - m) }
        val s = e.sum()
        for (i in e.indices) e[i] /= s
        return e
    }

    private fun dot(a: FloatArray, b: FloatArray): Float {
        var s = 0f
        for (i in a.indices) s += a[i] * b[i]
        return s
    }

    private fun readMatrix(arr: org.json.JSONArray): Array<FloatArray> =
        Array(arr.length()) { i ->
            val row = arr.getJSONArray(i)
            FloatArray(row.length()) { j -> row.getDouble(j).toFloat() }
        }

    companion object {
        private const val SIDE = 224
        private val MEAN = floatArrayOf(0.48145466f, 0.4578275f, 0.40821073f)
        private val STD = floatArrayOf(0.26862954f, 0.26130258f, 0.27577711f)
    }
}
