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
 */

import {css, html, LitElement, nothing} from 'lit';
import {customElement, query, state} from 'lit/decorators.js';

import {sharedStyles} from '../styles/shared_styles.js';

import {trapTabKey} from './focus_util.js';

/* tslint:disable:no-new-decorators */

function stopTracks(stream: MediaStream) {
  for (const track of stream.getTracks()) {
    track.stop();
  }
}

/**
 * A modal that previews the camera and takes a photo.
 *
 * Fires `open-change` (`{open}`) when it opens or closes, and
 * `photo-captured` (`{photo}`) with the JPEG photo.
 */
@customElement('litert-camera-capture')
export class LitertCameraCapture extends LitElement {
  static override styles = [
    sharedStyles, css`
      :host {
        display: contents;
      }

      .btn-icon {
        width: 34px;
        height: 34px;
        border-radius: 10px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: transparent;
        border: 1px solid transparent;
        color: var(--text-muted, #94a3b8);
        font-size: 0.95rem;
        font-family: inherit;
        cursor: pointer;
        transition: background-color 0.14s ease, color 0.14s ease, border-color 0.14s ease;
        flex-shrink: 0;
        padding: 0;
      }

      .btn-icon:hover:not(:disabled) {
        background: var(--bg-elevated, #1e2432);
        color: var(--text, #f1f5f9);
      }

      .modal-backdrop {
        position: fixed;
        inset: 0;
        background-color: var(--overlay-backdrop, rgba(4, 7, 13, 0.72));
        backdrop-filter: blur(6px);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 15000;
        padding: 20px;
      }

      .modal-card {
        background-color: var(--bg-card, #171c26);
        border: 1px solid var(--border, #262f40);
        border-radius: 16px;
        width: 100%;
        max-width: 500px;
        display: flex;
        flex-direction: column;
        overflow: hidden;
        box-shadow: var(--shadow-md, 0 10px 28px rgba(0, 0, 0, 0.48));
      }

      .modal-header {
        padding: 14px 18px;
        border-bottom: 1px solid var(--border, #262f40);
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-weight: 600;
        font-size: 0.92rem;
        color: var(--text, #f1f5f9);
      }

      .camera-preview-container {
        background: #000000;
        width: 100%;
        aspect-ratio: 4 / 3;
        display: flex;
        align-items: center;
        justify-content: center;
        overflow: hidden;
      }

      .camera-stream {
        width: 100%;
        height: 100%;
        object-fit: cover;
      }

      .modal-actions {
        padding: 14px 18px;
        display: flex;
        justify-content: flex-end;
        gap: 10px;
        border-top: 1px solid var(--border, #262f40);
      }
    `
  ];

  @state() private isOpen = false;
  @query('#camera-video') private video?: HTMLVideoElement;
  private stream: MediaStream|null = null;
  /** Incremented by each `open` and `close`, to drop superseded streams. */
  private openRequest = 0;

  /** Opens the camera. Resolves to false if it isn't available. */
  async open(): Promise<boolean> {
    if (!navigator.mediaDevices?.getUserMedia) return false;
    this.close();
    const request = ++this.openRequest;
    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: {facingMode: 'user', width: {ideal: 1280}, height: {ideal: 720}},
      });
    } catch (err) {
      if (request !== this.openRequest) return true;
      console.warn(
          '[LiteRT-LM] Camera stream failed, falling back to file picker:',
          err);
      return false;
    }
    if (request !== this.openRequest) {
      // Closed or reopened while waiting for camera permission.
      stopTracks(stream);
      return true;
    }
    this.stream = stream;
    this.setOpen(true);
    await this.updateComplete;
    if (this.video && this.stream) {
      this.video.srcObject = this.stream;
    }
    this.renderRoot.querySelector<HTMLElement>('#btn-capture-photo')?.focus();
    return true;
  }

  /** Stops the camera and closes the modal. */
  close() {
    this.openRequest++;
    if (this.stream) {
      stopTracks(this.stream);
    }
    this.stream = null;
    if (this.isOpen) {
      this.setOpen(false);
    }
  }

  override disconnectedCallback() {
    this.close();
    super.disconnectedCallback();
  }

  private setOpen(open: boolean) {
    this.isOpen = open;
    this.dispatchEvent(new CustomEvent('open-change', {detail: {open}}));
  }

  private handleKeyDown(e: KeyboardEvent) {
    if (e.key === 'Escape') {
      e.preventDefault();
      e.stopPropagation();
      this.close();
      return;
    }
    trapTabKey(e, e.currentTarget as HTMLElement);
  }

  private takePhoto() {
    const video = this.video;
    // Wait for the first frame rather than capturing a black image.
    if (!video?.videoWidth) return;
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      this.close();
      return;
    }
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob((photo) => {
      if (photo) {
        this.dispatchEvent(
            new CustomEvent('photo-captured', {detail: {photo}}));
      }
      this.close();
    }, 'image/jpeg', 0.92);
  }

  override render() {
    if (!this.isOpen) return nothing;
    return html`
      <div
        class="modal-backdrop"
        @click=${(e: Event) => {
          if (e.target === e.currentTarget) this.close();
        }}
        @keydown=${this.handleKeyDown}
      >
        <div
          class="modal-card"
          role="dialog"
          aria-modal="true"
          aria-labelledby="camera-modal-title"
        >
          <div class="modal-header">
            <span id="camera-modal-title">Take a photo</span>
            <button
              type="button"
              class="btn-icon"
              title="Close camera"
              aria-label="Close camera"
              @click=${this.close}
            >
              ✕
            </button>
          </div>
          <div class="camera-preview-container">
            <video
              id="camera-video"
              class="camera-stream"
              autoplay
              playsinline
            ></video>
          </div>
          <div class="modal-actions">
            <button
              type="button"
              class="btn btn-secondary"
              @click=${this.close}
            >
              Cancel
            </button>
            <button
              id="btn-capture-photo"
              type="button"
              class="btn btn-primary"
              @click=${this.takePhoto}
            >
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path>
                <circle cx="12" cy="13" r="4"></circle>
              </svg>
              <span>Take photo</span>
            </button>
          </div>
        </div>
      </div>
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'litert-camera-capture': LitertCameraCapture;
  }
}
