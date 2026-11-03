package com.vocavision.app

/** Class names and thresholds, copied from the desktop vision_engine.py so both apps agree. */
object Labels {
    private val RAW = arrayOf(
        "apple_pie", "fried_rice", "hamburger", "hot_dog", "ice_cream",
        "pizza", "ramen", "samosa", "sushi", "waffles",
        "t_shirt", "trouser", "pullover", "dress", "coat",
        "sandal", "shirt", "sneaker", "bag", "ankle_boot"
    )
    const val COUNT = 20
    private const val FIRST_CLOTHING = 10

    private const val DEFAULT_THRESHOLD = 0.52f
    private val THRESHOLDS = mapOf(18 to 0.75f, 17 to 0.70f, 15 to 0.65f, 19 to 0.65f)

    private val CLOTHING_NAMES = mapOf(
        "t_shirt" to "T-shirt", "trouser" to "trousers", "pullover" to "pullover",
        "dress" to "dress", "coat" to "coat", "sandal" to "sandals", "shirt" to "shirt",
        "sneaker" to "sneakers", "bag" to "bag", "ankle_boot" to "ankle boots"
    )

    /** Maps a CLIP label (e.g. "t-shirt", "ice cream", "boot") to a YOLO class id, or -1. */
    fun fromClip(name: String): Int {
        val raw = when (name) {
            "t-shirt" -> "t_shirt"
            "trousers" -> "trouser"
            "boot" -> "ankle_boot"
            else -> name.replace(' ', '_')
        }
        return RAW.indexOf(raw)
    }

    fun rawName(cls: Int) = RAW[cls]

    fun isUpperBody(cls: Int) = cls == 10 || cls == 16 || cls == 12 || cls == 13 || cls == 14
    const val TROUSER = 11

    fun threshold(cls: Int) = THRESHOLDS[cls] ?: DEFAULT_THRESHOLD

    fun isFood(cls: Int) = cls < FIRST_CLOTHING

    fun display(cls: Int): String {
        val raw = RAW[cls]
        return if (isFood(cls)) raw.split("_").joinToString(" ") { it.replaceFirstChar(Char::uppercase) }
        else CLOTHING_NAMES[raw] ?: raw
    }
}
