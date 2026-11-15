using System;
using System.Collections.Generic;
using System.Text;

namespace cstorchcl.tokenizers
{
    public class Tokenizer256
    {
        public Dictionary<string, int> vocabulary;
        public Dictionary<int, string> reverse_vocabulary;
        public int next_token { get; private set; }
        public Tokenizer256()
        {
            vocabulary = new();
            reverse_vocabulary = new Dictionary<int, string>();
            for (int i = 0; i < 256; i++)
            {
                char c = (char)i;
                vocabulary.Add(c.ToString(), i);
                reverse_vocabulary.Add(i, c.ToString());
            }
            next_token = 256;
        }
        public void load_from_json(string filepath)
        {
            using (StreamReader reader = new StreamReader(filepath))
            {
                while (reader.EndOfStream == false)
                {
                    string line = reader.ReadLine();
                    if (line != "{" && line != "}" && line.Length > 1)
                    {
                        string[] l = line.Replace('"'.ToString() + ",", "").Split('"'.ToString() + ":");
                        l[0] = l[0].Replace('"'.ToString(), "").Replace(" ", "");
                        l[1] = l[1].Replace('"'.ToString(), "");
                        l[1] = l[1].Substring(1, l[1].Length - 1);
                        //Console.WriteLine($"'{l[0]}':'{l[1]}'");
                        int token_id = int.Parse(l[0]);
                        string token = l[1];
                        vocabulary.Add(token, token_id);
                        reverse_vocabulary.Add(token_id, token);
                    }

                }
            }
        }

        private bool Contains(string value)
        {
            return vocabulary.ContainsKey(value);
        }
        public List<int> encode(string text)
        {
            List<int> encoded_tokens = new List<int>();
            int max_token_len = reverse_vocabulary[0].Length;
            int i = 0;
            for (i = 1; i < reverse_vocabulary.Count; i++)
                if (reverse_vocabulary[i].Length > max_token_len)
                    max_token_len = reverse_vocabulary[i].Length;
            i = 0;
            string current_value = "";
            UTF8Encoding uTF8Encoding = new UTF8Encoding();
            while (i < text.Length)
            {
                current_value = text.Substring(i, 1);
                if (!Contains(current_value))
                {
                    byte[] dat_ = uTF8Encoding.GetBytes(current_value);
                    for (int j = 0; j < dat_.Length; j++)
                        encoded_tokens.Add(dat_[j]);
                    i += 1;
                }
                else
                {
                    if (i + 1 < text.Length && "([{ ".Contains(text.Substring(i + 1, 1)) == false)
                    {
                        int j = i + 1;
                        int t = j;
                        while (t <= text.Length && t - i <= max_token_len)
                        {
                            if (Contains(text.Substring(i, t - i)))
                            {
                                j = t;
                            }
                            t++;
                        }
                        string tok_ = text.Substring(i, j - i);
                        encoded_tokens.Add(vocabulary[tok_]);
                        i += tok_.Length;

                    }
                    else
                    {
                        encoded_tokens.Add(vocabulary[current_value]);
                        i += 1;
                    }

                }

            }
            return encoded_tokens;
        }
        public List<int> encode(string text, int max_token_count)
        {
            List<int> encoded_tokens = new List<int>();
            int max_token_len = reverse_vocabulary[0].Length;
            int i = 0;
            for (i = 1; i < reverse_vocabulary.Count; i++)
                if (reverse_vocabulary[i].Length > max_token_len)
                    max_token_len = reverse_vocabulary[i].Length;
            i = 0;
            string current_value = "";
            UTF8Encoding uTF8Encoding = new UTF8Encoding();
            while (i < text.Length)
            {
                current_value = text.Substring(i, 1);
                if (!Contains(current_value))
                {
                    byte[] dat_ = uTF8Encoding.GetBytes(current_value);
                    if (encoded_tokens.Count + dat_.Length > max_token_count)
                        return encoded_tokens;
                    for (int j = 0; j < dat_.Length; j++)
                        encoded_tokens.Add(dat_[j]);
                    i += 1;
                }
                else
                {
                    if (i + 1 < text.Length && "([{ ".Contains(text.Substring(i + 1, 1)) == false)
                    {
                        int j = i + 1;
                        int t = j;
                        while (t <= text.Length && t - i <= max_token_len)
                        {
                            if (Contains(text.Substring(i, t - i)))
                            {
                                j = t;
                            }
                            t++;
                        }
                        string tok_ = text.Substring(i, j - i);
                        if (encoded_tokens.Count + 1 > max_token_count)
                            return encoded_tokens;
                        encoded_tokens.Add(vocabulary[tok_]);
                        i += tok_.Length;

                    }
                    else
                    {
                        if (encoded_tokens.Count + 1 > max_token_count)
                            return encoded_tokens;
                        encoded_tokens.Add(vocabulary[current_value]);
                        i += 1;
                    }

                }

            }
            return encoded_tokens;
        }
        
        public string decode(int[] array)
        {
            UTF8Encoding uTF8Encoding = new UTF8Encoding();

            StringBuilder builder = new StringBuilder();
            List<byte> current_val = new();
            for (int i = 0; i < array.Length; i++)
            {
                if (array[i] < 256)
                {
                    current_val.Add((byte)array[i]);
                }
                else
                {
                    if (current_val.Count > 0)
                    {
                        builder.Append(uTF8Encoding.GetString(current_val.ToArray()));
                        current_val.Clear();
                    }
                    builder.Append(reverse_vocabulary[array[i]]);
                }
            }
            if (current_val.Count > 0)
            {
                builder.Append(uTF8Encoding.GetString(current_val.ToArray()));
                current_val.Clear();
            }
            return builder.ToString();
        }
    }
}
