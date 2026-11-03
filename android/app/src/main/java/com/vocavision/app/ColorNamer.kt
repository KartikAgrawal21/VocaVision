package com.vocavision.app

import android.graphics.Bitmap
import android.graphics.Color

/** Names the dominant colour(s) of a garment, using the same HSV ranges as utils/color_detector.py. */
object ColorNamer {

    private class Range(val name: String, val h0: Int, val s0: Int, val v0: Int, val h1: Int, val s1: Int, val v1: Int)

    // OpenCV scale: H 0..180, S and V 0..255. Order matters: the first match wins.
    private val RANGES = listOf(
        Range("white", 0, 0, 210, 180, 30, 255),
        Range("black", 0, 0, 0, 180, 255, 40),
        Range("gray", 0, 0, 41, 180, 40, 209),
        Range("red", 0, 70, 50, 10, 255, 255),
        Range("red", 165, 70, 50, 180, 255, 255),
        Range("orange", 11, 80, 50, 25, 255, 255),
        Range("yellow", 26, 70, 50, 35, 255, 255),
        Range("beige", 0, 10, 180, 30, 60, 230),
        Range("brown", 8, 60, 20, 25, 220, 160),
        Range("green", 36, 50, 40, 85, 255, 255),
        Range("navy", 86, 80, 20, 130, 255, 80),
        Range("blue", 86, 50, 80, 130, 255, 255),
        Range("purple", 131, 50, 40, 160, 255, 255),
        Range("pink", 0, 20, 150, 10, 70, 255),
        Range("pink", 155, 30, 150, 180, 255, 255)
    )

    private const val GRID = 24
    private val hsv = FloatArray(3)

    /** Looks at the middle 60% of the box (less background) and returns e.g. "white" or "blue and white". */
    fun name(src: Bitmap, x1: Float, y1: Float, x2: Float, y2: Float): String? {
        val w = x2 - x1
        val h = y2 - y1
        val left = ((x1 + w * 0.2f) * src.width).toInt().coerceIn(0, src.width - 1)
        val top = ((y1 + h * 0.2f) * src.height).toInt().coerceIn(0, src.height - 1)
        val cw = (w * 0.6f * src.width).toInt().coerceIn(1, src.width - left)
        val ch = (h * 0.6f * src.height).toInt().coerceIn(1, src.height - top)
        if (cw < 8 || ch < 8) return null

        val small = Bitmap.createScaledBitmap(Bitmap.createBitmap(src, left, top, cw, ch), GRID, GRID, true)
        val votes = HashMap<String, Int>()
        for (y in 0 until GRID) for (x in 0 until GRID) {
            Color.colorToHSV(small.getPixel(x, y), hsv)
            val n = classify((hsv[0] / 2f).toInt(), (hsv[1] * 255f).toInt(), (hsv[2] * 255f).toInt()) ?: continue
            votes[n] = (votes[n] ?: 0) + 1
        }
        small.recycle()
        if (votes.isEmpty()) return null

        val total = GRID * GRID
        val ranked = votes.entries.sortedByDescending { it.value }
        val main = ranked.filter { it.value >= total * 0.2 }.take(2).map { it.key }
        return (main.ifEmpty { listOf(ranked.first().key) }).joinToString(" and ")
    }

    // Reference colours (sRGB) for the nearest-colour check; "beige" also covers skin tones.
    private val PALETTE = listOf(
        "white" to intArrayOf(240, 240, 240), "black" to intArrayOf(25, 25, 28), "gray" to intArrayOf(128, 128, 128),
        "red" to intArrayOf(200, 30, 40), "blue" to intArrayOf(40, 90, 200), "navy" to intArrayOf(20, 30, 90),
        "green" to intArrayOf(40, 150, 70), "yellow" to intArrayOf(235, 210, 50), "orange" to intArrayOf(240, 140, 30),
        "pink" to intArrayOf(240, 150, 180), "purple" to intArrayOf(120, 60, 160), "brown" to intArrayOf(110, 70, 40),
        "beige" to intArrayOf(215, 190, 155)
    )
    private val PALETTE_LAB = PALETTE.map { lab(it.second) }

    /** Per-channel median RGB of a region given as fractions of the image; ignores outlier pixels. */
    fun medianRgb(src: Bitmap, l: Float, t: Float, r: Float, b: Float): IntArray? {
        val x = (l * src.width).toInt().coerceIn(0, src.width - 2)
        val y = (t * src.height).toInt().coerceIn(0, src.height - 2)
        val w = ((r - l) * src.width).toInt().coerceIn(2, src.width - x)
        val h = ((b - t) * src.height).toInt().coerceIn(2, src.height - y)
        if (w < 8 || h < 8) return null
        val small = Bitmap.createScaledBitmap(Bitmap.createBitmap(src, x, y, w, h), 16, 16, true)
        val rs = IntArray(256); val gs = IntArray(256); val bs = IntArray(256)
        for (i in 0 until 256) {
            val p = small.getPixel(i % 16, i / 16)
            rs[i] = Color.red(p); gs[i] = Color.green(p); bs[i] = Color.blue(p)
        }
        small.recycle()
        rs.sort(); gs.sort(); bs.sort()
        return intArrayOf(rs[128], gs[128], bs[128])
    }

    /** Closest named colour to an RGB value, measured in CIELAB (matches how people see differences). */
    fun nearest(rgb: IntArray): String {
        val lab = lab(rgb)
        var best = 0
        var bestD = Double.MAX_VALUE
        for (i in PALETTE_LAB.indices) {
            val d = (lab[0] - PALETTE_LAB[i][0]).let { it * it } +
                (lab[1] - PALETTE_LAB[i][1]).let { it * it } + (lab[2] - PALETTE_LAB[i][2]).let { it * it }
            if (d < bestD) { bestD = d; best = i }
        }
        return PALETTE[best].first
    }

    fun saturation(rgb: IntArray): Float {
        val mx = maxOf(rgb[0], rgb[1], rgb[2]).toFloat()
        val mn = minOf(rgb[0], rgb[1], rgb[2]).toFloat()
        return if (mx == 0f) 0f else (mx - mn) / mx
    }

    private fun lab(rgb: IntArray): DoubleArray {
        fun lin(c: Int): Double { val v = c / 255.0; return if (v <= 0.04045) v / 12.92 else Math.pow((v + 0.055) / 1.055, 2.4) }
        val r = lin(rgb[0]); val g = lin(rgb[1]); val b = lin(rgb[2])
        val x = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047
        val y = 0.2126 * r + 0.7152 * g + 0.0722 * b
        val z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883
        fun f(t: Double) = if (t > 0.008856) Math.cbrt(t) else 7.787 * t + 16.0 / 116
        return doubleArrayOf(116 * f(y) - 16, 500 * (f(x) - f(y)), 200 * (f(y) - f(z)))
    }

    private fun classify(h: Int, s: Int, v: Int): String? {
        for (r in RANGES) {
            if (h in r.h0..r.h1 && s in r.s0..r.s1 && v in r.v0..r.v1) return r.name
        }
        return null
    }
}
