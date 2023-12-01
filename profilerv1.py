import tensorflow.compat.v1 as tf
import tensorflow.compat.v2 as tf2
tf.flags.DEFINE_string('graph_def', '/tmp/graphdef', 'graphdef path')
tf.flags.DEFINE_multi_string('input_names', [], 'input names')
tf.flags.DEFINE_multi_string('output_names', [], 'output names')
FLAGS = tf.flags.FLAGS

def profiler_for_graphdef(graph_def, input_names, output_names, inputs=None, logdir='/tmp/tensorboard', options=None):
    """profile a GraphDef with profiler v2 api.

    For example:

    >>> python3 profilerv1.py --graph_def=/tmp/graphdef --input_names=x:0 --input_names=y_input:0 --output_names=Mean:0


    Args:
        graph_def: A GraphDef or a GraphDef path.
        input_names: input tensor names.
        output_names: output tensor names.
        inputs: specific inputs, default is ones. (optional)
        logdir: tensorboard output directory, default is /tmp/tensorboard. (optional)
        options: ProfilerOptions, view https://www.tensorflow.org/api_docs/python/tf/profiler/experimental/ProfilerOptions. (optional)

    Returns:
        A list of Tensors, run results.

    """
    if isinstance(graph_def, str):
        graphdef = tf.GraphDef()
        with tf.io.gfile.GFile(graph_def, 'rb') as f:
            graphdef.ParseFromString(f.read())
    else:
        graphdef = graph_def
    imports_graph_def_fn = lambda: tf.import_graph_def(graphdef, name='')
    wrapped_import = tf.wrap_function(imports_graph_def_fn, [])
    import_graph = wrapped_import.graph
    input_tensors = tf.nest.map_structure(import_graph.as_graph_element, input_names)
    output_tensors = tf.nest.map_structure(import_graph.as_graph_element, output_names)
    if not inputs:
        inputs = [tf2.ones(shape=input_tensor.shape, dtype=input_tensor.dtype) for input_tensor in input_tensors]
    model = wrapped_import.prune(input_tensors, output_tensors)
    tf2.profiler.experimental.start(logdir, options=options)
    results = model(*inputs)
    tf2.profiler.experimental.stop()
    return results

if __name__ == '__main__':
    profiler_for_graphdef(FLAGS.graph_def, FLAGS.input_names, FLAGS.output_names)