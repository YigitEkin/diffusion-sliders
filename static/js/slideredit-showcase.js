document.addEventListener('DOMContentLoaded', function() {
  function ensureFrameLoaded(frame) {
    if (!frame) {
      return;
    }

    if (frame.tagName === 'IMG') {
      if (!frame.getAttribute('src') && frame.dataset.src) {
        frame.src = frame.dataset.src;
      }
      return;
    }

    if (frame.tagName === 'VIDEO') {
      var source = frame.querySelector('source');
      if (!source) {
        return;
      }

      var pendingSrc = source.dataset.src;
      if (pendingSrc && source.getAttribute('src') !== pendingSrc) {
        source.src = pendingSrc;
        frame.load();
      }
    }
  }

  document.querySelectorAll('.slider-card').forEach(function(card) {
    var input = card.querySelector('.showcase-range');
    var gallery = card.querySelector('[data-slider-gallery]');
    var frames = Array.prototype.slice.call(gallery.querySelectorAll('img, video'));
    var maxIndex = Math.max(frames.length - 1, 0);
    var activeVideoTime = 0;
    var activeIndex = 0;

    input.min = '0';
    input.max = String(maxIndex);
    input.step = '1';

    frames.forEach(function(frame) {
      if (frame.tagName === 'VIDEO') {
        frame.addEventListener('timeupdate', function() {
          if (frame.classList.contains('is-active')) {
            activeVideoTime = frame.currentTime;
          }
        });
      }
    });

    function render() {
      var index = Math.round(Number(input.value));
      var clampedIndex = Math.max(0, Math.min(index, maxIndex));
      var previousFrame = frames[activeIndex];
      var previousTime = activeVideoTime;

      if (previousFrame && previousFrame.tagName === 'VIDEO' && previousFrame.classList.contains('is-active')) {
        previousTime = previousFrame.currentTime;
      }

      frames.forEach(function(frame, frameIndex) {
        var isActive = frameIndex === clampedIndex;
        frame.classList.toggle('is-active', isActive);

        if (frame.tagName === 'VIDEO') {
          if (isActive) {
            ensureFrameLoaded(frame);
            var applyTime = function() {
              var duration = Number.isFinite(frame.duration) ? frame.duration : 0;
              frame.currentTime = duration > 0 ? Math.min(previousTime, Math.max(duration - 0.05, 0)) : previousTime;
              frame.play().catch(function() {});
            };

            if (frame.readyState >= 1) {
              applyTime();
            } else {
              frame.addEventListener('loadedmetadata', applyTime, { once: true });
              frame.load();
            }
          } else {
            frame.pause();
          }
        }
      });

      activeIndex = clampedIndex;
      activeVideoTime = previousTime;
    }

    input.addEventListener('input', render);
    render();
  });
});
