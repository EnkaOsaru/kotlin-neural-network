import kotlin.math.pow

class NeuralNetwork(
    private vararg val sizes: Int
) {

    private val weightCount = sizes.run {
        var result = 0
        for (i in 0 until lastIndex) {
            result += (this[i] + 1) * this[i + 1]
        }
        result
    }

    private val layers = Array(sizes.size - 1) { i ->
        Array(sizes[i] + 1) {
            DoubleArray(sizes[i + 1]) {
                Math.random()
            }
        }
    }
    private val activations = Array(sizes.size) { { v: Double -> if (v < 0) 0.0 else v } }
    private val error = { outputs: DoubleArray, answers: DoubleArray ->
        var result = 0.0
        for (i in outputs.indices) {
            result += (outputs[i] - answers[i]).pow(2)
        }
        result / outputs.size
    }
    private val learningRate = 0.01

    fun forward(inputs: DoubleArray): DoubleArray {
        var values = inputs
        layers.forEachIndexed { i, weightsArray ->
            val activation = activations[i]
            val newValues = DoubleArray(sizes[i + 1])
            weightsArray.forEachIndexed { j, weights ->
                weights.forEachIndexed { k, weight ->
                    newValues[k] += weight * if (values.size == j) 1.0 else activation.invoke(values[j])
                }
            }
            values = newValues
        }
        return values
    }

    fun train(inputsList: Array<DoubleArray>, answersList: Array<DoubleArray>, epochs: Int) {
        repeat(epochs) {
            for (i in inputsList.indices) {
                val inputs = inputsList[i]
                val answers = answersList[i]
                val flatWeights = DoubleArray(weightCount)
                var counter = 0
                layers.forEach { weightsList ->
                    weightsList.forEach { weights ->
                        for (j in weights.indices) {
                            val originalWeight = weights[j]
                            weights[j] -= DELTA
                            val e1 = error.invoke(forward(inputs), answers)
                            weights[j] = originalWeight + DELTA
                            val e2 = error.invoke(forward(inputs), answers)
                            weights[j] = originalWeight
                            val gradient = (e2 - e1) / DELTA
                            flatWeights[counter++] = originalWeight - learningRate * gradient
                        }
                    }
                }
                counter = 0
                layers.forEach { weightsList ->
                    weightsList.forEach { weights ->
                        for (j in weights.indices) {
                            weights[j] = flatWeights[counter++]
                        }
                    }
                }
            }
        }
    }

    companion object {

        private const val DELTA = 0.00005
    }
}

fun main() {
    val inputsList = arrayOf(
        doubleArrayOf(0.0, 0.0, 0.0), doubleArrayOf(0.0, 0.0, 1.0), doubleArrayOf(0.0, 1.0, 0.0),
        doubleArrayOf(0.0, 1.0, 1.0), doubleArrayOf(1.0, 0.0, 0.0), doubleArrayOf(1.0, 0.0, 1.0),
        doubleArrayOf(1.0, 1.0, 0.0), doubleArrayOf(1.0, 1.0, 1.0)
    )
    val outputsList = arrayOf(
        doubleArrayOf(0.0, 0.0), doubleArrayOf(0.0, 1.0), doubleArrayOf(0.0, 1.0),
        doubleArrayOf(1.0, 0.0), doubleArrayOf(0.0, 1.0), doubleArrayOf(1.0, 0.0),
        doubleArrayOf(1.0, 0.0), doubleArrayOf(1.0, 1.0)
    )
    val nn = NeuralNetwork(3, 5, 2)
    nn.train(inputsList, outputsList, 2000)

    for (inputs in inputsList) {
        val outputs = nn.forward(inputs)
        print(inputs.contentToString() + " -> ")
        println(outputs.contentToString())
    }
}
