package com.vocavision.app

import android.graphics.Bitmap

/** A detection plus the human label shown on screen and spoken aloud. */
class Found(val det: Det, val label: String, val isFood: Boolean)

class Description(val sentence: String, val found: List<Found>, val note: String = "") {
    val isEmpty get() = found.isEmpty()
}

/**
 * Combines YOLO and CLIP the same way the desktop vision_engine.py does:
 *  - YOLO finds nothing           -> CLIP names the main item
 *  - YOLO finds one weak/odd item -> CLIP double-checks it and may override
 *  - YOLO finds several things    -> keep YOLO's multi-object result (shirt and trousers)
 */
class Pipeline(private val detector: Detector, private val clip: ClipClassifier?) {

    private val notes = ArrayList<String>()

    fun run(frame: Bitmap): Description {
        notes.clear()
        val d = runInternal(frame)
        return Description(d.sentence, d.found, notes.joinToString("\n"))
    }

    private fun runInternal(frame: Bitmap): Description {
        val all = detector.detectAll(frame, YOLO_SIZE)
        val dets = all.strict

        if (dets.isEmpty()) {
            return if (clip == null) Describer.sentence(emptyList()) else fromClip(frame, clip.classify(frame), all.loose)
        }

        if (clip != null) {
            val best = dets.maxByOrNull { it.conf }!!
            val strong = dets.filter { it.conf >= 0.75f }.map { it.cls }.toSet()
            if (strong.size < 2) {
                val guess = clip.classify(frame)
                val a = norm(guess.name)
                val b = norm(Labels.rawName(best.cls))
                val agrees = a in b || b in a
                val clipWins = guess.kind != "none" && !agrees &&
                    (guess.conf >= 0.85f || (best.conf < 0.60f && guess.conf >= best.conf))
                if (clipWins) return fromClip(frame, guess, all.loose)
            }
        }
        return fromYolo(frame, dets)
    }

    private fun fromYolo(frame: Bitmap, dets: List<Det>): Description {
        val found = dets.map { d ->
            val food = Labels.isFood(d.cls)
            val name = Labels.display(d.cls)
            val colour = if (food) null else colourOfBox(frame, d)
            Found(d, if (colour != null) "$colour $name" else name, food)
        }
        return Describer.sentence(found)
    }

    private fun colourOfBox(frame: Bitmap, d: Det): String? {
        val w = d.x2 - d.x1
        val h = d.y2 - d.y1
        if (w > 0.05f && h > 0.05f) {
            return colourFor(frame, d.x1 + w * 0.2f, d.y1 + h * 0.2f, d.x2 - w * 0.2f, d.y2 - h * 0.2f)
        }
        return ColorNamer.name(frame, d.x1, d.y1, d.x2, d.y2)
    }

    /**
     * Colour of a region, measured from its pixels (hue first, then dark/light). CLIP is only asked
     * when the pixels say "light gray", which in dim light can really be a white item.
     */
    private fun colourFor(frame: Bitmap, l: Float, t: Float, r: Float, b: Float): String? {
        val px = ColorNamer.region(frame, l, t, r, b)
        if (px == null) {
            return try { clip?.colourOf(frame, l, t, r, b) } catch (_: Throwable) { null }
        }
        var name = px.name
        var note = "pixels=${px.name} (${(px.share * 100).toInt()}% of area, S=${"%.2f".format(px.saturation)}, V=${"%.2f".format(px.value)})"
        if (px.maybeWhite && clip != null) {
            val cl = try { clip.colourOf(frame, l, t, r, b) } catch (_: Throwable) { null }
            note += " clip=$cl"
            // One direction only: CLIP calls a white striped shirt "gray", so it may upgrade to white
            // (and only for a bright result), never downgrade a white.
            if (cl == "white" && px.value >= 0.70f) name = "white"
        }
        notes.add("colour: $note -> $name")
        return name
    }

    /** Builds the result from CLIP alone: the top item, plus shirt AND trousers only when both are proven. */
    private fun fromClip(frame: Bitmap, g: ClipClassifier.Guess, loose: List<Det>): Description {
        val c = clip!!
        val unsure = "I am not sure what this is. Please move closer and try again."
        if (g.kind == "none" || g.conf < 0.30f) return Description(unsure, emptyList())
        val topCls = Labels.fromClip(g.name)
        if (topCls < 0) return Description(unsure, emptyList())

        val split = if (Labels.isUpperBody(topCls) || topCls == Labels.TROUSER) splitCheck(frame, c) else null

        // Where is the garment? A weak YOLO box of the same kind is a good locator even when it is too
        // unsure to be reported; without one, fall back to the middle of the photo (never the bottom,
        // which is trousers, belt or floor).
        fun locate(cls: Int): FloatArray? {
            val d = loose.filter {
                sameKind(it.cls, cls) && it.conf >= 0.20f && it.x2 - it.x1 >= 0.15f && it.y2 - it.y1 >= 0.15f
            }.maxByOrNull { it.conf } ?: return null
            return floatArrayOf(d.x1, d.y1, d.x2, d.y2)
        }

        val items = ArrayList<Triple<Int, Float, FloatArray>>()  // class, confidence, box
        if (split != null) {
            items.add(Triple(split.upper, split.upperConf, locate(split.upper) ?: floatArrayOf(0.05f, 0.05f, 0.95f, 0.55f)))
            items.add(Triple(split.lower, split.lowerConf, locate(split.lower) ?: floatArrayOf(0.05f, 0.45f, 0.95f, 0.97f)))
        } else {
            items.add(Triple(topCls, g.conf, locate(topCls) ?: floatArrayOf(0.05f, 0.05f, 0.95f, 0.95f)))
        }

        val found = items.map { (cls, p, box) ->
            val food = Labels.isFood(cls)
            val name = Labels.display(cls)
            val label = if (food) name else {
                val located = locate(cls) != null
                val colour = try {
                    when {
                        located -> {
                            val w = box[2] - box[0]
                            val h = box[3] - box[1]
                            colourFor(frame, box[0] + w * 0.2f, box[1] + h * 0.2f, box[2] - w * 0.2f, box[3] - h * 0.2f)
                        }
                        split != null && Labels.isUpperBody(cls) -> colourFor(frame, 0.25f, 0.12f, 0.75f, 0.42f)
                        split != null -> colourFor(frame, 0.30f, 0.62f, 0.70f, 0.92f)
                        else -> colourFor(frame, 0.30f, 0.28f, 0.70f, 0.65f)
                    }
                } catch (_: Throwable) { null }
                if (colour != null) "$colour $name" else name
            }
            Found(Det(cls, p, box[0], box[1], box[2], box[3]), label, food)
        }
        return Describer.sentence(found)
    }

    private class Split(val upper: Int, val upperConf: Float, val lower: Int, val lowerConf: Float)

    /**
     * Shirt and trousers together are only reported when CLIP, looking at each half of the photo on its own,
     * sees a shirt-type garment in the top half and trousers in the bottom half. A lone T-shirt lying on a
     * floor looks like a T-shirt in both halves, so it is not given invented trousers.
     */
    private fun splitCheck(frame: Bitmap, c: ClipClassifier): Split? {
        val top = c.classify(c.cropOf(frame, 0f, 0f, 1f, 0.55f))
        val bottom = c.classify(c.cropOf(frame, 0f, 0.45f, 1f, 1f))
        val up = Labels.fromClip(top.name)
        val low = Labels.fromClip(bottom.name)
        notes.add("halves: top=${top.name} ${(top.conf * 100).toInt()}%, bottom=${bottom.name} ${(bottom.conf * 100).toInt()}%")
        val both = up >= 0 && Labels.isUpperBody(up) && top.conf >= 0.35f &&
            low == Labels.TROUSER && bottom.conf >= 0.50f
        return if (both) Split(up, top.conf, low, bottom.conf) else null
    }

    /** Same kind of thing: the identical class, or both are tops. */
    private fun sameKind(a: Int, b: Int) = a == b || (Labels.isUpperBody(a) && Labels.isUpperBody(b))

    private fun norm(s: String) = s.lowercase().filter { it.isLetter() }

    companion object {
        private const val YOLO_SIZE = 640
    }
}

/** Builds the spoken sentence, mirroring _build_description in the desktop vision_engine.py. */
object Describer {

    fun sentence(found: List<Found>): Description {
        val foods = found.filter { it.isFood }
        val clothes = found.filter { !it.isFood }

        val parts = ArrayList<String>()
        if (foods.isNotEmpty()) parts.add("I can see ${oxford(counted(foods))}.")
        if (clothes.isNotEmpty()) {
            val lead = if (parts.isEmpty()) "I can see" else "I can also see"
            parts.add("$lead ${oxford(counted(clothes))}.")
        }
        val text = if (parts.isEmpty()) "I could not detect any known objects." else parts.joinToString(" ")
        return Description(text, found)
    }

    private fun counted(items: List<Found>): List<String> =
        items.groupingBy { it.label }.eachCount().map { (label, n) ->
            when {
                n > 1 -> "$n ${plural(label)}"
                isPlural(label) -> label                       // "beige trousers", "white sneakers"
                else -> "${article(label)} $label"             // "a T-shirt", "an orange T-shirt"
            }
        }

    // "dress" ends in s but is singular; "trousers", "sandals" and "waffles" are plural.
    private fun isPlural(label: String) = label.endsWith("s") && !label.endsWith("ss")

    private fun article(label: String) = if (label.first().lowercaseChar() in "aeiou") "an" else "a"

    private fun plural(label: String) = when {
        isPlural(label) -> label
        label.endsWith("ss") -> label + "es"
        else -> label + "s"
    }

    private fun oxford(items: List<String>): String = when (items.size) {
        1 -> items[0]
        2 -> "${items[0]} and ${items[1]}"
        else -> items.dropLast(1).joinToString(", ") + ", and " + items.last()
    }
}
