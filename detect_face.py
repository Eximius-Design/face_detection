def detect(img,sess, graph, min_size,factor,thresholds):
	feeds = {graph.get_operation_by_name('input').outputs[0]: img,graph.get_operation_by_name('min_size').outputs[0]: min_size,graph.get_operation_by_name('thresholds').outputs[0]: thresholds,graph.get_operation_by_name('factor').outputs[0]: factor}
	fetches = [graph.get_operation_by_name('prob').outputs[0], graph.get_operation_by_name('landmarks').outputs[0], graph.get_operation_by_name('box').outputs[0]]
	prob, landmarks, box = sess.run(fetches, feeds)
	return box, prob, landmarks
