
<img src="images/1.jpg" height="400" />

---

## Text to Vector

### /api/v1/text2vec

input: single word for now (can add skip-thoughts to embed sentences at a later stage)

example curl call:

```curl -F 'text=cat' ec2-34-243-15-197.eu-west-1.compute.amazonaws.com:8787/api/v1/text2vec```

---

## Image to Vector

### /api/v1/image2vec

example curl call:

```curl -F 'image=@/Users/roelof/Desktop/animals-cool-funny-squirrel-wallpaper-hd-widescreen-wallpaper.jpg' ec2-34-243-15-197.eu-west-1.compute.amazonaws.com:8787/api/v1/image2vec```

---

## Image Path

### /api/v1/path

json input: v1 and v2 (vector/list)

example curl call:

```curl -X POST -H "Content-Type: application/json" -d '{"v1":[long_vector],"v2":[long_vector]}' ec2-34-243-15-197.eu-west-1.compute.amazonaws.com:8787/api/v1/path```
or
```curl -H "Content-Type: application/json" --data @data.json ec2-34-243-15-197.eu-west-1.compute.amazonaws.com:8787/api/v1/path```

response:
```{"status": "ok", "response": "0|static/serve/b6284dc8728b4b5c8c30867775b751d3.jpg|0||1|static/serve/13623f530b384c4c99f7f1aadb9ddd12.jpg|0.730695366859||2|static/serve/b1626ba43f1b4a0a970d6c7ca7eeb0c2.jpg|0.743275284767||3|static/serve/af6754e82b404ca99b57d941643b4929.jpg|0.825378775597||4|static/serve/b31e931b67924ff4922841cd9f2ca9a8.jpg|0.831463754177||5|static/serve/63b960bc745644dd855e6b4ca89380b1.jpg|1||"}```

---

## Image to Captions (retrieve)

<img src="images/2.gif" height="400" />

### /api/v1/image2caption

example curl call:

```curl -F 'image=@/Users/roelof/Downloads/5503072658_00e4af05ac_z.jpg' ec2-34-243-15-197.eu-west-1.compute.amazonaws.com:8787/api/v1/image2caption```

response:

```{"status": "ok", "response": "[\"BLURRED MOVEMENT OF TWO PEOPLE RUNNING ON THE BEACH HOLDING A SURF BOARD\", \"A yellow car travels down the road with a surf board on top .\", \"A man with a surfboard walking across the beach at sunset .\", \"Person on a beach watching a single-sail boat out past the surf at sunset .\", \"A person stands beside a surfboard on the beach just after sunrise\"]"}```

---

## Image to Story (generate)

<img src="images/3.gif" height="400" />

### /api/v1/image2story

example curl call:

```curl -F 'image=@/Users/roelof/Downloads/5503072658_00e4af05ac_z.jpg' ec2-34-243-15-197.eu-west-1.compute.amazonaws.com:8787/api/v1/image2story```

response:

```{"status": "ok", "response": "A man wearing a beach house on the horizon ."}```

---

## Image to Romance (generate)

<img src="images/4.gif" height="400" />

### /api/v1/image2romance

example curl call:

```curl -F 'image=@/Users/roelof/Downloads/5503072658_00e4af05ac_z.jpg' ec2-34-243-15-197.eu-west-1.compute.amazonaws.com:8787/api/v1/image2romance```

response:

```{"status": "ok", "response": "My descent was at the beach , and I closed my eyes for a moment . For the first time in years , I had no idea who he really wanted to be . The sun rose up on the horizon , making it appear as if she 'd fallen asleep on the beach . Her boyfriend had saved her life , and that was the most beautiful thing in the world . I didn t know what to do , so I let go of Alex s arm and pull him out of the water . The waves seemed to fade , leaving me at the conclusion I was beautiful ."}```

---

## Image to Regularities (pos/neg) (retrieve)

<img src="images/5.gif" height="400" />

### /api/v1/image2reg

example curl call:

```curl -F 'pos=sea' -F 'neg=cloud' -F 'image=@/Users/roelof/Downloads/5503072658_00e4af05ac_z.jpg' ec2-34-243-15-197.eu-west-1.compute.amazonaws.com:8787/api/v1/image2reg```

response:

```{"status": "ok", "response": "[\"A person with a surf board in the sea\", \"a person is out at sea carrying a surf board\", \"There is a ocean and a individual carrying a surfboard .\", \"A shirtless man carrying a surfboard out of the water\", \"A motor boat moving fast in the sea\"]"}```
