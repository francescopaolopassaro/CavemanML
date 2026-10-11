// ------------------------------------------------------------------------------
// CavemanML
// Copyright (c) 2026 Passaro Francesco Paolo
// Licensed under the MIT License. See LICENSE file in the project root.
//
// Dependency-free readers for the two weight formats the Synthelion models ship in: safetensors (a JSON
// header followed by raw tensor bytes) and the PyTorch ".bin" checkpoint (a zip holding a pickled state_dict
// and one raw file per tensor storage). No torch, no onnxruntime, no native code.
// ------------------------------------------------------------------------------
using System.Buffers.Binary;
using System.IO.Compression;
using System.Text;
using System.Text.Json;

namespace Caveman.ML.Weights;

/// <summary>A dense float32 tensor in row-major order.</summary>
public sealed class Tensor
{
    public int[] Shape { get; }
    public float[] Data { get; }

    public Tensor(int[] shape, float[] data)
    {
        Shape = shape;
        Data = data;
    }

    public int Rank => Shape.Length;
    public int Length => Data.Length;
    public override string ToString() => $"Tensor[{string.Join("x", Shape)}]";
}

/// <summary>Reads model weights from <c>model.safetensors</c> or a PyTorch <c>model.bin</c> checkpoint.</summary>
public static class TensorFiles
{
    /// <summary>Loads every tensor of the file, keyed by parameter name. The format is chosen from the extension.</summary>
    public static Dictionary<string, Tensor> Load(string path)
    {
        var ext = Path.GetExtension(path).ToLowerInvariant();
        return ext == ".safetensors" ? LoadSafeTensors(path) : LoadTorchCheckpoint(path);
    }

    // ───────────────────────────── safetensors ─────────────────────────────

    public static Dictionary<string, Tensor> LoadSafeTensors(string path)
    {
        var bytes = File.ReadAllBytes(path);
        var headerLen = (long)BinaryPrimitives.ReadUInt64LittleEndian(bytes.AsSpan(0, 8));
        using var doc = JsonDocument.Parse(new ReadOnlyMemory<byte>(bytes, 8, (int)headerLen));
        var dataStart = 8 + (int)headerLen;
        var result = new Dictionary<string, Tensor>(StringComparer.Ordinal);

        foreach (var entry in doc.RootElement.EnumerateObject())
        {
            if (entry.Name == "__metadata__") continue;
            var dtype = entry.Value.GetProperty("dtype").GetString();
            var shape = entry.Value.GetProperty("shape").EnumerateArray().Select(e => e.GetInt32()).ToArray();
            var offsets = entry.Value.GetProperty("data_offsets").EnumerateArray().Select(e => e.GetInt64()).ToArray();
            var count = shape.Aggregate(1, (a, b) => a * b);
            var span = bytes.AsSpan(dataStart + (int)offsets[0], (int)(offsets[1] - offsets[0]));
            result[entry.Name.Replace('/', '.')] = new Tensor(shape, ToFloats(span, dtype!, count));   // the HF export uses "/" where torch uses "."
        }
        return result;
    }

    private static float[] ToFloats(ReadOnlySpan<byte> raw, string dtype, int count)
    {
        var data = new float[count];
        switch (dtype)
        {
            case "F32":
                for (var i = 0; i < count; i++) data[i] = BinaryPrimitives.ReadSingleLittleEndian(raw.Slice(i * 4, 4));
                break;
            case "F16":
                for (var i = 0; i < count; i++) data[i] = (float)BinaryPrimitives.ReadHalfLittleEndian(raw.Slice(i * 2, 2));
                break;
            case "BF16":
                for (var i = 0; i < count; i++) data[i] = BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadUInt16LittleEndian(raw.Slice(i * 2, 2)) << 16);
                break;
            default:
                throw new NotSupportedException($"safetensors dtype '{dtype}' is not supported (expected F32, F16 or BF16).");
        }
        return data;
    }

    // ───────────────────────────── PyTorch checkpoint ─────────────────────────────

    public static Dictionary<string, Tensor> LoadTorchCheckpoint(string path)
    {
        using var zip = ZipFile.OpenRead(path);
        var pickleEntry = zip.Entries.FirstOrDefault(e => e.FullName.EndsWith("/data.pkl", StringComparison.Ordinal) || e.FullName == "data.pkl")
            ?? throw new InvalidDataException("not a PyTorch checkpoint: no data.pkl entry");
        var prefix = pickleEntry.FullName[..^"data.pkl".Length];

        byte[] pickleBytes;
        using (var s = pickleEntry.Open()) { using var ms = new MemoryStream(); s.CopyTo(ms); pickleBytes = ms.ToArray(); }

        var storages = new Dictionary<string, byte[]>();
        byte[] LoadStorage(string key)
        {
            if (storages.TryGetValue(key, out var cached)) return cached;
            var entry = zip.GetEntry(prefix + "data/" + key) ?? throw new InvalidDataException($"missing tensor storage '{key}'");
            using var s = entry.Open();
            using var ms = new MemoryStream();
            s.CopyTo(ms);
            return storages[key] = ms.ToArray();
        }

        var vm = new PickleReader(pickleBytes, LoadStorage);
        var root = vm.Run();
        var result = new Dictionary<string, Tensor>(StringComparer.Ordinal);
        if (root is not IDictionary<object, object?> dict)
            throw new InvalidDataException("checkpoint root is not a state_dict");
        foreach (var kv in dict)
            if (kv.Value is PickleReader.TensorRef t && kv.Key is string name)
                result[name] = t.ToTensor();
        return result;
    }

    /// <summary>
    /// A minimal pickle virtual machine - just enough of protocol 2-5 to read a PyTorch state_dict:
    /// an OrderedDict of <c>torch._utils._rebuild_tensor_v2(storage, offset, size, stride, ...)</c> calls whose
    /// storages are persistent ids resolved from the zip.
    /// </summary>
    internal sealed class PickleReader
    {
        public sealed record StorageRef(string Key, string DType, int Numel, Func<string, byte[]> Load);

        public sealed class TensorRef
        {
            public StorageRef Storage = null!;
            public long Offset;
            public int[] Shape = Array.Empty<int>();
            public long[] Stride = Array.Empty<long>();

            public Tensor ToTensor()
            {
                var count = Shape.Aggregate(1, (a, b) => a * b);
                var raw = Storage.Load(Storage.Key);
                var width = Storage.DType switch { "FloatStorage" => 4, "HalfStorage" or "BFloat16Storage" => 2, _ => throw new NotSupportedException($"storage type {Storage.DType}") };
                var data = new float[count];
                // contiguous only (the case for every parameter in these checkpoints)
                var start = (int)Offset * width;
                for (var i = 0; i < count; i++)
                {
                    var o = start + i * width;
                    data[i] = Storage.DType switch
                    {
                        "FloatStorage" => BinaryPrimitives.ReadSingleLittleEndian(raw.AsSpan(o, 4)),
                        "HalfStorage" => (float)BinaryPrimitives.ReadHalfLittleEndian(raw.AsSpan(o, 2)),
                        _ => BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadUInt16LittleEndian(raw.AsSpan(o, 2)) << 16),
                    };
                }
                return new Tensor(Shape, data);
            }
        }

        private sealed class Mark { }
        private sealed record Global(string Module, string Name);

        private readonly byte[] _b;
        private readonly Func<string, byte[]> _loadStorage;
        private int _p;
        private readonly List<object?> _stack = new();
        private readonly List<int> _marks = new();
        private readonly Dictionary<int, object?> _memo = new();

        public PickleReader(byte[] data, Func<string, byte[]> loadStorage)
        {
            _b = data;
            _loadStorage = loadStorage;
        }

        private static readonly Mark MarkObj = new();

        private void Push(object? v) => _stack.Add(v);
        private object? Pop() { var v = _stack[^1]; _stack.RemoveAt(_stack.Count - 1); return v; }

        private List<object?> PopToMark()
        {
            var m = _marks[^1]; _marks.RemoveAt(_marks.Count - 1);
            var items = _stack.GetRange(m, _stack.Count - m);
            _stack.RemoveRange(m, _stack.Count - m);
            return items;
        }

        private byte U8() => _b[_p++];
        private int U16() { var v = BinaryPrimitives.ReadUInt16LittleEndian(_b.AsSpan(_p)); _p += 2; return v; }
        private int I32() { var v = BinaryPrimitives.ReadInt32LittleEndian(_b.AsSpan(_p)); _p += 4; return v; }
        private uint U32() { var v = BinaryPrimitives.ReadUInt32LittleEndian(_b.AsSpan(_p)); _p += 4; return v; }
        private ulong U64() { var v = BinaryPrimitives.ReadUInt64LittleEndian(_b.AsSpan(_p)); _p += 8; return v; }

        private string ReadLine()
        {
            var s = _p;
            while (_b[_p] != (byte)'\n') _p++;
            var line = Encoding.ASCII.GetString(_b, s, _p - s);
            _p++;
            return line;
        }

        private string Utf8(int len) { var s = Encoding.UTF8.GetString(_b, _p, len); _p += len; return s; }

        public object? Run()
        {
            while (true)
            {
                var op = U8();
                switch (op)
                {
                    case 0x80: _p++; break;                                 // PROTO
                    case 0x95: _p += 8; break;                              // FRAME
                    case (byte)'.': return Pop();                           // STOP
                    case (byte)'(': _marks.Add(_stack.Count); break;        // MARK
                    case (byte)'0': Pop(); break;                           // POP
                    case (byte)'N': Push(null); break;
                    case 0x88: Push(true); break;
                    case 0x89: Push(false); break;
                    case (byte)'K': Push((long)U8()); break;                // BININT1
                    case (byte)'M': Push((long)U16()); break;               // BININT2
                    case (byte)'J': Push((long)I32()); break;               // BININT
                    case 0x8A: { var n = U8(); long v = 0; for (var i = 0; i < n; i++) v |= (long)_b[_p + i] << (8 * i); if (n > 0 && (_b[_p + n - 1] & 0x80) != 0 && n < 8) v -= 1L << (8 * n); _p += n; Push(v); break; } // LONG1
                    case (byte)'G': { var v = BinaryPrimitives.ReadDoubleBigEndian(_b.AsSpan(_p)); _p += 8; Push(v); break; } // BINFLOAT
                    case (byte)'X': Push(Utf8((int)U32())); break;          // BINUNICODE
                    case 0x8C: Push(Utf8(U8())); break;                     // SHORT_BINUNICODE
                    case 0x8D: Push(Utf8((int)U64())); break;               // BINUNICODE8
                    case (byte)'U': Push(Utf8(U8())); break;                // SHORT_BINSTRING
                    case (byte)'T': Push(Utf8((int)U32())); break;          // BINSTRING
                    case (byte)')': Push(Array.Empty<object?>()); break;    // EMPTY_TUPLE
                    case 0x85: { var a = Pop(); Push(new[] { a }); break; } // TUPLE1
                    case 0x86: { var b = Pop(); var a = Pop(); Push(new[] { a, b }); break; }
                    case 0x87: { var c = Pop(); var b = Pop(); var a = Pop(); Push(new[] { a, b, c }); break; }
                    case (byte)'t': Push(PopToMark().ToArray()); break;     // TUPLE
                    case (byte)']': Push(new List<object?>()); break;       // EMPTY_LIST
                    case (byte)'}': Push(new Dictionary<object, object?>()); break; // EMPTY_DICT
                    case (byte)'a': { var v = Pop(); ((List<object?>)_stack[^1]!).Add(v); break; } // APPEND
                    case (byte)'e': { var items = PopToMark(); ((List<object?>)_stack[^1]!).AddRange(items); break; } // APPENDS
                    case (byte)'s': { var v = Pop(); var k = Pop(); ((IDictionary<object, object?>)_stack[^1]!)[k!] = v; break; } // SETITEM
                    case (byte)'u': { var items = PopToMark(); var d = (IDictionary<object, object?>)_stack[^1]!; for (var i = 0; i + 1 < items.Count; i += 2) d[items[i]!] = items[i + 1]; break; } // SETITEMS
                    case 0x94: _memo[_memo.Count] = _stack[^1]; break;      // MEMOIZE
                    case (byte)'q': _memo[U8()] = _stack[^1]; break;        // BINPUT
                    case (byte)'r': _memo[(int)U32()] = _stack[^1]; break;  // LONG_BINPUT
                    case (byte)'h': Push(_memo[U8()]); break;               // BINGET
                    case (byte)'j': Push(_memo[(int)U32()]); break;         // LONG_BINGET
                    case (byte)'c': { var module = ReadLine(); var name = ReadLine(); Push(new Global(module, name)); break; } // GLOBAL
                    case 0x93: { var name = (string)Pop()!; var module = (string)Pop()!; Push(new Global(module, name)); break; } // STACK_GLOBAL
                    case (byte)'R': { var args = (object?[])Pop()!; var fn = Pop(); Push(Call(fn, args)); break; } // REDUCE
                    case 0x81: { var args = (object?[])Pop()!; var cls = Pop(); Push(Call(cls, args)); break; }   // NEWOBJ
                    case (byte)'b': { Pop(); break; }                       // BUILD: state is ignored (OrderedDict metadata)
                    case (byte)'Q': { var pid = (object?[])Pop()!; Push(PersistentLoad(pid)); break; } // BINPERSID
                    default: throw new InvalidDataException($"unsupported pickle opcode 0x{op:X2} at {_p - 1}");
                }
            }
        }

        private object? PersistentLoad(object?[] pid)
        {
            // ('storage', <storage type>, key, location, numel)
            if (pid.Length >= 5 && pid[0] is string tag && tag == "storage")
            {
                var type = pid[1] is Global g ? g.Name : pid[1]?.ToString() ?? "";
                return new StorageRef((string)pid[2]!, type, Convert.ToInt32(pid[4]), _loadStorage);
            }
            throw new InvalidDataException("unsupported persistent id");
        }

        private object? Call(object? fn, object?[] args)
        {
            if (fn is Global g)
            {
                switch (g.Module + "." + g.Name)
                {
                    case "collections.OrderedDict": return new Dictionary<object, object?>();
                    case "torch._utils._rebuild_tensor_v2":
                    {
                        var t = new TensorRef
                        {
                            Storage = (StorageRef)args[0]!,
                            Offset = Convert.ToInt64(args[1]),
                            Shape = ((object?[])args[2]!).Select(Convert.ToInt32).ToArray(),
                            Stride = ((object?[])args[3]!).Select(Convert.ToInt64).ToArray(),
                        };
                        return t;
                    }
                    case "torch._utils._rebuild_parameter": return args[0];
                    default:
                        if (g.Name.EndsWith("Storage", StringComparison.Ordinal)) return g;
                        throw new InvalidDataException($"unsupported pickle callable {g.Module}.{g.Name}");
                }
            }
            throw new InvalidDataException("pickle REDUCE on a non-callable");
        }
    }
}
