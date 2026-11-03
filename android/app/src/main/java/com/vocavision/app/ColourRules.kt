package com.vocavision.app

import kotlin.math.max
import kotlin.math.min

/**
 * Names the dominant colour of a patch of pixels. Pure Kotlin (no Android classes) so it can be unit tested.
 *
 * Why not "nearest reference colour": dark green is closer to black than to bright green, and pale blue is
 * closer to gray than to saturated blue. So the order here is: neutral (black/gray/white) or coloured first,
 * then the hue family, then dark/light from the brightness.
 */
object ColourRules {

    class Result(
        val name: String,
        val share: Float,          // fraction of the patch that belongs to the winning colour
        val saturation: Float,     // median saturation (0..1) of the winning group
        val value: Float,          // median brightness (0..1) of the winning group
        val maybeWhite: Boolean    // light gray that could really be a white item in dim light
    )

    private const val BINS = 24    // 15 degrees each

    fun analyse(argb: IntArray): Result? {
        if (argb.isEmpty()) return null
        val n = argb.size
        val hue = FloatArray(n)
        val sat = FloatArray(n)
        val value = FloatArray(n)
        val neutral = BooleanArray(n)
        val bins = IntArray(BINS)
        var neutralCount = 0

        for (i in 0 until n) {
            val p = argb[i]
            val r = ((p shr 16) and 0xFF) / 255f
            val g = ((p shr 8) and 0xFF) / 255f
            val b = (p and 0xFF) / 255f
            val mx = max(r, max(g, b))
            val mn = min(r, min(g, b))
            val d = mx - mn
            val s = if (mx == 0f) 0f else d / mx
            var h = 0f
            if (d > 0f) {
                h = when (mx) {
                    r -> 60f * (((g - b) / d) % 6f)
                    g -> 60f * ((b - r) / d + 2f)
                    else -> 60f * ((r - g) / d + 4f)
                }
                if (h < 0f) h += 360f
            }
            hue[i] = h; sat[i] = s; value[i] = mx
            // Near-black, or hardly any colour, counts as neutral.
            val isNeutral = mx < 0.17f || s < 0.11f || (s < 0.30f && mx < 0.26f)
            neutral[i] = isNeutral
            if (isNeutral) neutralCount++ else bins[(h / 15f).toInt().coerceIn(0, BINS - 1)]++
        }

        // The strongest 45-degree window of hues, so a colour on a family boundary is not split in two.
        var centre = 0
        var windowSum = -1
        for (c in 0 until BINS) {
            val sum = bins[(c + BINS - 1) % BINS] + bins[c] + bins[(c + 1) % BINS]
            if (sum > windowSum) { windowSum = sum; centre = c }
        }

        if (neutralCount >= windowSum) {
            val vs = FloatArray(neutralCount)
            val ss = FloatArray(neutralCount)
            var k = 0
            for (i in 0 until n) if (neutral[i]) { vs[k] = value[i]; ss[k] = sat[i]; k++ }
            val v = median(vs)
            val name = neutralName(v)
            return Result(name, neutralCount / n.toFloat(), median(ss), v, name == "light gray")
        }

        val centreHue = centre * 15f + 7.5f
        val offsets = ArrayList<Float>()
        val ss = ArrayList<Float>()
        val vs = ArrayList<Float>()
        for (i in 0 until n) {
            if (neutral[i]) continue
            val bin = (hue[i] / 15f).toInt().coerceIn(0, BINS - 1)
            val dist = min((bin - centre + BINS) % BINS, (centre - bin + BINS) % BINS)
            if (dist > 1) continue
            offsets.add(((hue[i] - centreHue + 540f) % 360f) - 180f)
            ss.add(sat[i]); vs.add(value[i])
        }
        val h = ((centreHue + median(offsets.toFloatArray()) + 360f) % 360f)
        val s = median(ss.toFloatArray())
        val v = median(vs.toFloatArray())
        return Result(colourName(h, s, v), offsets.size / n.toFloat(), s, v, false)
    }

    private fun neutralName(v: Float) = when {
        v >= 0.80f -> "white"
        v >= 0.62f -> "light gray"
        v >= 0.40f -> "gray"
        v >= 0.20f -> "dark gray"
        else -> "black"
    }

    private fun colourName(h: Float, s: Float, v: Float): String = when {
        h < 15f || h >= 345f -> when {
            v < 0.50f -> "maroon"
            s < 0.45f && v > 0.65f -> "pink"
            else -> "red"
        }
        h < 45f -> when {           // orange, brown and beige/skin tones share this hue
            s < 0.50f -> if (v < 0.45f) "brown" else "beige"
            else -> if (v < 0.70f || s < 0.70f) "brown" else "orange"
        }
        h < 70f -> when {
            v < 0.55f -> "olive"
            s < 0.45f -> "beige"
            else -> "yellow"
        }
        h < 165f -> when {
            h < 90f && v < 0.55f -> "olive"
            v < 0.40f -> "dark green"
            s < 0.35f && v > 0.75f -> "light green"
            else -> "green"
        }
        h < 190f -> if (v < 0.40f) "dark teal" else "teal"
        h < 255f -> when {
            h >= 225f && v < 0.40f -> "navy"
            v < 0.40f -> "dark blue"
            s < 0.50f && v > 0.70f -> "light blue"
            else -> "blue"
        }
        h < 300f -> when {
            v < 0.40f -> "dark purple"
            s < 0.40f && v > 0.70f -> "lavender"
            else -> "purple"
        }
        else -> if (v < 0.40f) "dark purple" else "pink"
    }

    private fun median(a: FloatArray): Float {
        if (a.isEmpty()) return 0f
        val s = a.sortedArray()
        return s[s.size / 2]
    }
}
