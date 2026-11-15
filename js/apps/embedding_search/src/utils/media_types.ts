/**
 * Copyright 2026 The ODML Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * @fileoverview Single source of truth for which files the demo can embed,
 * shared by the directory scanner, drag-and-drop, and the file picker.
 */

import {SUPPORTED_AUDIO_MIME_TYPES, SUPPORTED_IMAGE_MIME_TYPES} from '@litert-lm/core';

/** The modality of a file, which determines how it is embedded. */
export type FileCategory = 'text'|'image'|'audio';

/** A file's category together with the MIME type to hand to the engine. */
export interface ClassifiedFile {
  category: FileCategory;
  mimeType: string;
}

const TEXT_EXTENSIONS = new Set([
  '.txt', '.md', '.markdown', '.json', '.csv', '.tsv', '.html', '.css',
  '.js', '.ts', '.jsx', '.tsx', '.py', '.c', '.cc', '.cpp', '.h', '.hpp',
  '.java', '.go', '.rs', '.yaml', '.yml', '.sh', '.bash', '.xml', '.sql',
  '.log', '.ini', '.toml', '.proto', '.rst', '.tex',
]);

const IMAGE_EXTENSIONS_MAP: Record<string, string> = {
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.png': 'image/png',
  '.bmp': 'image/bmp',
  '.gif': 'image/gif',
  '.webp': 'image/webp',
  '.avif': 'image/avif',
  '.svg': 'image/svg+xml',
  '.tga': 'image/x-tga',
  '.ppm': 'image/x-portable-pixmap',
  '.pgm': 'image/x-portable-graymap',
};

const AUDIO_EXTENSIONS_MAP: Record<string, string> = {
  '.wav': 'audio/wav',
  '.wave': 'audio/wave',
  '.mp3': 'audio/mp3',
  '.flac': 'audio/flac',
};

/** Returns the lower-cased extension of `name` including the dot, or ''. */
export function fileExtension(name: string): string {
  const base = name.slice(name.lastIndexOf('/') + 1);
  const dot = base.lastIndexOf('.');
  return dot <= 0 ? '' : base.slice(dot).toLowerCase();
}

/**
 * Classifies a file by its name alone, for use while walking a directory
 * where only the entry name is known. Returns null for unsupported files.
 */
export function classifyFileName(name: string): ClassifiedFile|null {
  const ext = fileExtension(name);
  if (TEXT_EXTENSIONS.has(ext)) return {category: 'text', mimeType: 'text/plain'};
  const image = IMAGE_EXTENSIONS_MAP[ext];
  if (image) return {category: 'image', mimeType: image};
  const audio = AUDIO_EXTENSIONS_MAP[ext];
  if (audio) return {category: 'audio', mimeType: audio};
  return null;
}

/**
 * Classifies a media file the user supplied (dropped, picked, or recorded)
 * as an image or audio clip, or null if the engine cannot embed it.
 */
export function classifyMediaFile(file: {name: string; type: string}):
    'image'|'audio'|null {
  const mime = file.type.toLowerCase();
  if (SUPPORTED_IMAGE_MIME_TYPES.has(mime) || mime.startsWith('image/')) {
    return 'image';
  }
  if (SUPPORTED_AUDIO_MIME_TYPES.has(mime)) return 'audio';
  const byName = classifyFileName(file.name);
  if (byName?.category === 'image' || byName?.category === 'audio') {
    return byName.category;
  }
  return null;
}

/**
 * Picks the MIME type to label `file`'s bytes with before embedding: the
 * browser's type if the engine recognises it, otherwise `fallback`.
 */
export function engineMimeType(file: {type: string}, fallback: string): string {
  const mime = file.type.toLowerCase();
  if (SUPPORTED_IMAGE_MIME_TYPES.has(mime) ||
      SUPPORTED_AUDIO_MIME_TYPES.has(mime)) {
    return mime;
  }
  return fallback;
}

/**
 * Ensures `blob` is in a format `EmbeddingEngine` natively decodes. Formats
 * the browser supports but `stb_image` does not (such as WebP, AVIF, or SVG)
 * are transcoded to PNG via `createImageBitmap` + `OffscreenCanvas`.
 */
export async function toSupportedMediaBlob(
    blob: Blob, category: 'image'|'audio', fallbackMime = ''): Promise<Blob> {
  const mime = engineMimeType(
      blob, fallbackMime || (category === 'image' ? 'image/png' : 'audio/wav'));
  if (category === 'audio' || SUPPORTED_IMAGE_MIME_TYPES.has(mime)) {
    return blob.type === mime ? blob : new Blob([await blob.arrayBuffer()], {type: mime});
  }
  if (typeof createImageBitmap === 'function' &&
      typeof OffscreenCanvas !== 'undefined') {
    const bitmap = await createImageBitmap(blob);
    try {
      const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.drawImage(bitmap, 0, 0);
        return await canvas.convertToBlob({type: 'image/png'});
      }
    } finally {
      bitmap.close();
    }
  }
  return new Blob([await blob.arrayBuffer()], {type: 'image/png'});
}

/**
 * Extracts dropped media `File`s from a `DataTransfer`, including images
 * dragged directly from web pages (which often populate `text/html` or
 * `text/uri-list` rather than `dataTransfer.files`).
 */
export async function extractDroppedFiles(dt: DataTransfer): Promise<File[]> {
  const html = dt.getData?.('text/html') ?? '';
  if (dt.files.length > 0) {
    const fromImgTag = /<img\b/i.test(html);
    return Array.from(dt.files, (f) => {
      if (!classifyMediaFile(f) && fromImgTag) {
        return new File([f], f.name || 'dropped-image.png', {type: 'image/png'});
      }
      return f;
    });
  }

  let url = '';
  const match =
      html.match(/<img\b[^>]*?\bsrc\s*=\s*(?:"([^"]+)"|'([^']+)'|([^\s>]+))/i);
  if (match) {
    url = (match[1] || match[2] || match[3] || '').replace(/&amp;/g, '&');
  }
  if (!url) {
    const uriList =
        dt.getData?.('text/uri-list') || dt.getData?.('text/plain') || '';
    const line =
        uriList.split(/\r?\n/).find((l) => l && !l.startsWith('#'))?.trim() ??
        '';
    if (/^(https?:|data:image\/|blob:)/i.test(line)) url = line;
  }
  if (!url) return [];

  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const blob = await res.blob();
  const rawName = url.startsWith('data:') ?
      'dropped-image.png' :
      decodeURIComponent(
          url.split('?')[0]!.split('/').pop() || 'dropped-image');
  const byName = classifyFileName(rawName);
  const mime = blob.type.toLowerCase();
  const type = (mime.startsWith('image/') || mime.startsWith('audio/')) ?
      mime :
      (byName?.mimeType || 'image/png');
  const name = fileExtension(rawName) ? rawName : `${rawName}.png`;
  return [new File([blob], name, {type})];
}
