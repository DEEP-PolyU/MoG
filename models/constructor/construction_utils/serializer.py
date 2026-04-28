
class GraphSerializer:

    @staticmethod
    def make_json_serializable(obj):
        try:
            import numpy as np
            numpy_available = True
        except ImportError:
            numpy_available = False

        if obj is None:
            return None
        elif isinstance(obj, (str, bool)):
            return obj
        elif isinstance(obj, (int, float)) and not (numpy_available and hasattr(obj, 'dtype')):
            return obj

        if numpy_available:
            obj_type_name = type(obj).__name__
            if any(numpy_type in obj_type_name for numpy_type in
                   ['int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64']):
                return int(obj)
            elif any(numpy_type in obj_type_name for numpy_type in ['float16', 'float32', 'float64', 'float128']):
                return float(obj)
            elif 'bool_' in obj_type_name or obj_type_name == 'bool_':
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return [GraphSerializer.make_json_serializable(item) for item in obj.tolist()]
            elif hasattr(np, 'integer') and isinstance(obj, np.integer):
                return int(obj)
            elif hasattr(np, 'floating') and isinstance(obj, np.floating):
                return float(obj)
            elif hasattr(np, 'bool_') and isinstance(obj, np.bool_):
                return bool(obj)
            elif hasattr(obj, 'dtype'):
                try:
                    if hasattr(obj, 'item'):
                        converted = obj.item()
                        return GraphSerializer.make_json_serializable(converted)
                    else:
                        return str(obj)
                except (ValueError, TypeError, AttributeError):
                    return str(obj)

        if isinstance(obj, dict):
            return {GraphSerializer.make_json_serializable(k): GraphSerializer.make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [GraphSerializer.make_json_serializable(item) for item in obj]
        elif isinstance(obj, set):
            return [GraphSerializer.make_json_serializable(item) for item in obj]

        elif hasattr(obj, '__dict__') and not isinstance(obj, type):
            return str(obj)
        else:
            try:
                if hasattr(obj, '__int__'):
                    return int(obj)
                elif hasattr(obj, '__float__'):
                    return float(obj)
                elif hasattr(obj, '__str__'):
                    return str(obj)
                else:
                    return str(obj)
            except (ValueError, TypeError, AttributeError):
                return str(obj)

    @staticmethod
    def deep_serialize_for_json(obj):
        if isinstance(obj, dict):
            return {str(k): GraphSerializer.deep_serialize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [GraphSerializer.deep_serialize_for_json(item) for item in obj]
        elif isinstance(obj, set):
            return list(GraphSerializer.deep_serialize_for_json(item) for item in obj)
        elif obj is None or isinstance(obj, (str, bool)):
            return obj
        elif isinstance(obj, (int, float)):
            return obj
        else:
            try:
                if hasattr(obj, 'item') and callable(getattr(obj, 'item', None)):
                    return GraphSerializer.deep_serialize_for_json(obj.item())
                elif hasattr(obj, 'tolist') and callable(getattr(obj, 'tolist', None)):
                    return GraphSerializer.deep_serialize_for_json(obj.tolist())
                elif hasattr(obj, '__int__') and not isinstance(obj, (dict, list, tuple, set)):
                    return int(obj)
                elif hasattr(obj, '__float__') and not isinstance(obj, (dict, list, tuple, set)):
                    return float(obj)
                else:
                    return str(obj)
            except (ValueError, TypeError, AttributeError):
                return str(obj)

