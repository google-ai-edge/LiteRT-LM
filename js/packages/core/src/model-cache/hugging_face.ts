/**
 * Copyright 2026 Google LLC
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
 */

import {getCacheEnvironment} from './environment.js';
import {ModelSource} from './model_source.js';

/**
 * A model file on the Hugging Face Hub, as
 * `hf://<org>/<repo>[@<revision>][/<path>]`, like `huggingface_hub`'s
 * `HfFileSystem` uses. A revision containing `/` must be percent-encoded, e.g.
 * `hf://org/repo@feature%2Fx`. Like `HfFileSystem`, pull request and
 * conversion refs (`refs/pr/<n>` and `refs/convert/<name>`) can also be given
 * unencoded, e.g. `hf://org/repo@refs/pr/1/model.litertlm`.
 */
export type HuggingFaceUri = `hf://${string}/${string}`;

/** The parts of a `HuggingFaceUri`. */
export declare interface HuggingFaceModelRef {
  /** The model repo, as `<org>/<repo>`. */
  repo: `${string}/${string}`;
  /** A branch, tag or commit. Defaults to 'main'. */
  revision?: string;
  /**
   * The model file's path in the repo. If omitted, the file is chosen with
   * `ResolveHuggingFaceModelOptions.selectFile`.
   */
  path?: string;
}

const SCHEME = 'hf://';
/** Revisions that may contain unencoded `/`s, as in `HfFileSystem`. */
const SPECIAL_REVISION = /^refs\/(pr\/\d+|convert\/\w+)(?=\/|$)/;

/**
 * Parses a `hf://` URI, returning undefined if it isn't one. Only model repos
 * are supported, so `hf://datasets/...` and `hf://spaces/...` aren't.
 */
export function parseHuggingFaceUri(uri: string): HuggingFaceModelRef|
    undefined {
  if (!uri.startsWith(SCHEME)) return undefined;
  const match = /^([^/@]+)\/([^/@]+)(@?)(.*)$/.exec(uri.slice(SCHEME.length));
  if (!match) return undefined;
  const [, org, name, at] = match;
  // Empty, or the path with a leading `/`, after the revision is removed.
  let rest = match[4];
  if (org === 'datasets' || org === 'spaces') return undefined;
  const ref: HuggingFaceModelRef = {repo: `${org}/${name}`};
  if (at) {
    const encoded = SPECIAL_REVISION.exec(rest)?.[0] ?? rest.split('/')[0];
    rest = rest.slice(encoded.length);
    let revision: string;
    try {
      revision = decodeURIComponent(encoded);
    } catch {
      return undefined;
    }
    if (!revision) return undefined;
    ref.revision = revision;
  }
  const path = rest.slice(1);
  if (path) ref.path = path;
  return ref;
}

/** Builds a `hf://` URI, percent-encoding the revision. */
export function toHuggingFaceUri(ref: HuggingFaceModelRef): HuggingFaceUri {
  const revision =
      ref.revision === undefined ? '' : `@${encodeURIComponent(ref.revision)}`;
  const path = ref.path ? `/${ref.path}` : '';
  return `${SCHEME}${ref.repo}${revision}${path}`;
}

/** A file in a Hugging Face model repo. */
export declare interface HuggingFaceFile {
  /** The file's path in the repo. */
  path: string;
  size: number;
  /** SHA-256 of the file as lowercase hex, for files stored with Git LFS. */
  sha256?: string;
}

/** Options for `resolveHuggingFaceModel()`. */
export declare interface ResolveHuggingFaceModelOptions {
  /**
   * The Hugging Face access token of the user loading the model, for gated or
   * private repos, e.g. from Sign in with Hugging Face (OAuth) with the
   * `gated-repos` scope. Don't embed your own token in a web page: everyone
   * who loads the page can read it. An empty token is the same as none.
   */
  accessToken?: string;
  /**
   * Chooses the model file when the URI doesn't give a path, from the files
   * at the top level of the repo at `revision`. Returns undefined if none
   * match.
   */
  selectFile?: (files: HuggingFaceFile[], revision: string) =>
      HuggingFaceFile | undefined;
  /**
   * Revisions to try, in order, when the URI doesn't give one. The first that
   * exists is used, even if it doesn't have the file the URI gives (or that
   * `selectFile` chooses), in which case resolving fails. Defaults to
   * `['main']`.
   */
  defaultRevisions?: string[];
  signal?: AbortSignal;
}

/** A model file on the Hugging Face Hub, pinned to a commit. */
export declare interface ResolvedHuggingFaceModel {
  /**
   * Where to download the model from, with its SHA-256 and size, and the
   * access token if one was given.
   */
  source: ModelSource;
  repo: `${string}/${string}`;
  /** The revision that was asked for, e.g. a branch. */
  revision: string;
  /** The commit `revision` resolved to, which `source` downloads from. */
  commit: string;
  /** The model file's path in the repo. */
  path: string;
}

const ENDPOINT = 'https://huggingface.co';

/**
 * Resolves a model on the Hugging Face Hub to a file at a specific commit.
 *
 * The revision is pinned to its current commit, so that the download URL and
 * the file's SHA-256 always match, and a branch that moves resolves to the new
 * file. Throws before anything is downloaded if the revision doesn't exist,
 * the repo isn't accessible, or no file matches.
 */
export async function resolveHuggingFaceModel(
    model: string|HuggingFaceModelRef,
    options: ResolveHuggingFaceModelOptions = {}):
    Promise<ResolvedHuggingFaceModel> {
  const ref = typeof model === 'string' ? parseHuggingFaceUri(model) : model;
  if (!ref) {
    throw new Error(`Not a Hugging Face model URI: ${model}`);
  }
  const {repo, path} = ref;
  const revisions = ref.revision !== undefined ?
      [ref.revision] :
      options.defaultRevisions ?? ['main'];
  if (revisions.length === 0) {
    throw new Error('defaultRevisions must not be empty');
  }

  // Look up all the revisions at once, and use the first that exists. Each
  // lookup is awaited in order, so this returns as soon as the first that
  // exists is known, without waiting for the rest.
  const lookups =
      revisions.map(revision => getRevision(repo, revision, path, options));
  for (const lookup of lookups) {
    // Lookups after the one that's used are never awaited.
    void lookup.catch(() => {});
  }
  let revision: string|undefined;
  let found: RevisionFiles|undefined;
  for (const [i, lookup] of lookups.entries()) {
    found = await lookup;
    if (found !== undefined) {
      revision = revisions[i];
      break;
    }
  }
  if (revision === undefined || found === undefined) {
    throw new Error(`Couldn't resolve ${
        toHuggingFaceUri({repo, path})}: no revision named ${
        revisions.map(r => `'${r}'`).join(' or ')} exists`);
  }
  const {commit, files} = found;
  const uri = toHuggingFaceUri({repo, revision, path});

  let file: HuggingFaceFile|undefined;
  if (path) {
    file = files.find(f => f.path === path);
    if (!file) {
      throw new Error(`${uri} doesn't exist`);
    }
  } else {
    if (!options.selectFile) {
      throw new Error(`${uri} doesn't give a file path`);
    }
    const topLevelFiles = files.filter(f => !f.path.includes('/'));
    file = options.selectFile(topLevelFiles, revision);
    if (!file) {
      throw new Error(`No suitable model file in ${uri}. Its files are: ${
          topLevelFiles.map(f => f.path).join(', ')}`);
    }
  }

  const source: ModelSource = {
    url: `${ENDPOINT}/${encodePath(repo)}/resolve/${commit}/${
        encodePath(file.path)}`,
    size: file.size,
  };
  if (file.sha256 !== undefined) source.sha256 = file.sha256;
  if (options.accessToken) {
    source.accessToken = options.accessToken;
  }
  return {source, repo, revision, commit, path: file.path};
}

/** The fields used from a file's Git LFS info in the Hub API. */
export declare interface HubFileLfs {
  sha256?: string;
  size?: number;
}

/** The fields used from a file in the Hub API's model revision info. */
export declare interface HubFile {
  /** The file's path in the repo. */
  rfilename?: string;
  size?: number;
  lfs?: HubFileLfs;
}

/** The fields used from the Hub API's model revision info. */
export declare interface HubRevisionInfo {
  sha?: string;
  siblings?: HubFile[];
}

declare interface RevisionFiles {
  commit: string;
  /** Every file in the repo at `commit`. */
  files: HuggingFaceFile[];
}

/**
 * Returns the commit that a revision points to and the files in the repo at
 * that commit, or undefined if the repo has no such revision.
 *
 * `blobs=true` adds each file's size and Git LFS info. The listing isn't
 * paginated, unlike the tree API's, and model repos are small.
 */
async function getRevision(
    repo: `${string}/${string}`, revision: string, path: string|undefined,
    options: ResolveHuggingFaceModelOptions):
    Promise<RevisionFiles|undefined> {
  const uri = toHuggingFaceUri({repo, revision, path});
  const response = await fetchHub(
      `${ENDPOINT}/api/models/${encodePath(repo)}/revision/${
          encodeURIComponent(revision)}?blobs=true`,
      options);
  if (response.status === 404 &&
      response.headers.get('X-Error-Code') === 'RevisionNotFound') {
    return undefined;
  }
  checkResponse(response, uri);
  const info = await response.json() as HubRevisionInfo;
  if (!info.sha) {
    throw new Error(`The Hugging Face Hub didn't return a commit for ${uri}`);
  }
  const files: HuggingFaceFile[] = [];
  for (const sibling of info.siblings ?? []) {
    if (sibling.rfilename === undefined) continue;
    const file: HuggingFaceFile = {
      path: sibling.rfilename,
      size: sibling.lfs?.size ?? sibling.size ?? 0,
    };
    if (sibling.lfs?.sha256 !== undefined) file.sha256 = sibling.lfs.sha256;
    files.push(file);
  }
  return {commit: info.sha, files};
}

/** Fetches from the Hub API, with the access token if there is one. */
async function fetchHub(
    url: string, options: ResolveHuggingFaceModelOptions): Promise<Response> {
  const init: RequestInit = {};
  if (options.signal) init.signal = options.signal;
  if (options.accessToken) {
    init.headers = {'Authorization': `Bearer ${options.accessToken}`};
  }
  return getCacheEnvironment().fetch(url, init);
}

/** Throws a helpful error if a Hub API request failed. */
function checkResponse(response: Response, uri: string): void {
  if (response.ok) return;
  const message = response.headers.get('X-Error-Message') ??
      `${response.status} ${response.statusText}`;
  if (response.status === 401 || response.status === 403) {
    throw new Error(
        `Couldn't access ${uri}: ${message}. The repo may not exist, or it ` +
        'may be private or gated, which needs the access token of a user ' +
        'who can access it (e.g. from Sign in with Hugging Face).');
  }
  throw new Error(`Couldn't resolve ${uri}: ${message}`);
}

/** Percent-encodes each segment of a path. */
function encodePath(path: string): string {
  return path.split('/').map(encodeURIComponent).join('/');
}
